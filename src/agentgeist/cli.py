import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .budget import BudgetExceeded, Ledger
from .contracts import Tick, safety_floor, team_handle
from .store import Store, process_lock


def utf8_console():
    """Windows pipes default to cp1252, which cannot encode CrewAI's console box drawing."""
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")


def emit(event, **data):
    print(json.dumps({"event": event, **data}, default=str), flush=True)


def auth(team):
    result = {"team": team}
    if os.getenv("SCENARIO_TOKEN"):
        result["token"] = os.environ["SCENARIO_TOKEN"]
    return result


def submit(scenario, store, team, tick, result):
    # Persist the exact action before the network call. Retries are server-idempotent per tick.
    store.update(team, tick.tick_id, "decision", result)
    from .connections import MCPToolError
    try:
        judgment = scenario.call("hack26-submit-containment", **auth(team),
                                 tick_id=tick.tick_id, action=result["decision"]["action"])
    except MCPToolError:
        store.status(team, tick.tick_id, "rejected")
        emit("submission_rejected", tick_id=tick.tick_id, resolution="Use resolve after inspecting the venue response")
        raise
    store.save_raw(team, {"tool": "hack26-submit-containment", "tick_id": tick.tick_id, "response": judgment})
    if isinstance(judgment, dict) and isinstance(judgment.get("judgment"), dict):
        judgment = judgment["judgment"]
    if (not isinstance(judgment, dict) or type(judgment.get("score")) not in (int, float)
            or not 0 <= judgment["score"] <= 1):
        store.status(team, tick.tick_id, "receipt_unverified")
        emit("receipt_unverified", tick_id=tick.tick_id, resolution="Inspect the saved raw response before retrying")
        raise RuntimeError("Unexpected judgment format; retain outbox for inspection")
    store.update(team, tick.tick_id, "judgment", judgment)
    emit("judgment", tick_id=tick.tick_id, judgment=judgment)
    return judgment


def run(args):
    from .connections import MCPConnection
    from .swarm import run_swarm, warm_up
    from .notifier import notify
    team = team_handle(args.team)
    store = Store(args.db)
    warm_up()  # before any tick is pulled: the decision clock then measures only the tick
    scenario = MCPConnection(os.environ["SCENARIO_MCP"])
    try:
        # Resume durable inbox/outbox first; never pull over an unfinished tick.
        for row in store.pending(team):
            if row["escalation"] and json.loads(row["escalation"]).get("status") == "unknown":
                emit("escalation_unknown", tick_id=row["tick_id"], resolution="Use resolve after checking Discord")
                continue
            if not row["decision"]:
                process_tick(Tick.model_validate_json(row["tick"]), args, team, store, scenario, run_swarm, notify)
            elif args.submit:
                tick = Tick.model_validate_json(row["tick"])
                result = json.loads(row["decision"])
                judgment = json.loads(row["judgment"]) if row["judgment"] else submit(scenario, store, team, tick, result)
                escalate(tick, result, judgment, args, team, store, notify)
        if not args.submit and any(not row["judgment"] for row in store.pending(team)):
            emit("submission_pending", resolution="Use run --submit to replay the saved decision")
            return
        for _ in range(args.ticks):
            data = scenario.call("hack26-get-tick", **auth(team), scenario=args.scenario)
            receipt_id = store.save_raw(team, data)
            if not isinstance(data, dict):
                raise ValueError("Invalid feed payload; raw ingress saved")
            if data.get("exhausted"):
                emit("feed_exhausted")
                break
            try:
                tick = Tick.model_validate(data.get("tick", data))
            except ValueError:
                emit("invalid_tick_saved", receipt_id=receipt_id)
                raise
            existing = store.get(team, tick.tick_id)
            if existing:
                emit("tick_already_saved", tick_id=tick.tick_id, status=existing["status"])
                continue
            store.save_tick(team, tick)
            process_tick(tick, args, team, store, scenario, run_swarm, notify)
            if not args.submit:
                break  # Leave one unsubmitted decision in the durable outbox.
        emit("scoreboard", data=scenario.call("hack26-get-scoreboard"))
    finally:
        scenario.close()


def process_tick(tick, args, team, store, scenario, run_swarm, notify):
    start = time.monotonic()
    ledger = tick_ledger(tick, team, store)
    try:
        result = run_swarm(tick, ledger)
    except Exception as exc:
        cause, visited = exc, set()
        budget_failure = False
        while cause is not None and id(cause) not in visited:
            visited.add(id(cause))
            budget_failure |= isinstance(cause, BudgetExceeded)
            cause = cause.__cause__ or cause.__context__
        if not budget_failure:
            store.status(team, tick.tick_id, "failed")
            from .connections import describe_error
            emit("tick_failed", tick_id=tick.tick_id, error=type(exc).__name__,
                 **{k: v for k, v in describe_error(exc).items() if k != "type"},
                 resolution="Use resolve to retry or abandon")
            raise
        result = {"decision": safety_floor(tick).model_dump(), "mode": "budget_fallback",
                  "model_decision_valid": False, "evidence": {}, "peak_active_agents": ledger.peak,
                  "credits_reserved": ledger.spent, "credits_remaining": ledger.remaining}
    result["decision_seconds"] = round(time.monotonic() - start, 3)
    result["credits_reserved"] = store.spent(team, tick.tick_id)
    store.update(team, tick.tick_id, "decision", result)
    emit("decision", tick_id=tick.tick_id, **{k:v for k,v in result.items() if k != "evidence"})
    if args.submit:
        judgment = submit(scenario, store, team, tick, result)
        escalate(tick, result, judgment, args, team, store, notify, ledger)
        if result.get("log_dir"):
            from .trace import record_outcome
            row = store.get(team, tick.tick_id)
            record_outcome(result["log_dir"], team=team, judgment=judgment,
                           escalation=json.loads(row["escalation"]) if row and row["escalation"] else None)
    if result.get("log_dir"):
        # After submission and Discord: the email report never delays a scored answer.
        from .report import email_report
        report = email_report(result["log_dir"])
        if report:
            emit("email_report", tick_id=tick.tick_id, status=report["status"], via=report["via"])


def escalate(tick, result, judgment, args, team, store, notify, ledger=None):
    row = store.get(team, tick.tick_id)
    prior = json.loads(row["escalation"]) if row and row["escalation"] else {}
    if prior.get("status") in ("sent", "user_confirmed", "unknown"):
        emit("escalation_skipped", tick_id=tick.tick_id, status=prior["status"])
        return
    if result["decision"]["action"] == "observe":
        store.update(team, tick.tick_id, "escalation", {"status": "not_required"})
        return
    if not args.discord or getattr(args, "discord_blocked", False):
        emit("escalation_pending", tick_id=tick.tick_id, reason="Enable --discord and configure the Discord transport")
        return
    ledger = ledger or tick_ledger(tick, team, store)
    store.update(team, tick.tick_id, "escalation", {"status": "pending"})
    try:
        receipt = notify(team, tick, result, judgment, ledger,
                         before_send=lambda: store.update(team, tick.tick_id, "escalation", {"status": "unknown"}),
                         transport=getattr(args, "discord_transport", None))
        store.update(team, tick.tick_id, "escalation", receipt)
        emit("discord_receipt", tick_id=tick.tick_id, receipt=receipt)
    except Exception as exc:
        args.discord_blocked = True
        # A Discord failure must not lose an accepted containment judgment.
        emit("escalation_pending", tick_id=tick.tick_id, error=type(exc).__name__)
        # The outbox remains pending or unknown. Containment processing can continue.


def tick_ledger(tick, team, store):
    return Ledger(tick.swarm_budget.max_active_agents,
                  max(0, min(tick.swarm_budget.api_credits_remaining - store.spent(team, tick.tick_id),
                             store.remaining(team))),
                  on_reserve=lambda units: store.reserve(team, tick.tick_id, units))


def main():
    utf8_console()
    load_dotenv()
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    parser = argparse.ArgumentParser(description="Agentgeist supervised containment swarm")
    parser.add_argument("--team", default=os.getenv("TEAM_HANDLE", "agentgeist"))
    parser.add_argument("--db", default="state/agentgeist.sqlite3")
    commands = parser.add_subparsers(dest="command", required=True)
    offline = commands.add_parser("demo", help="Offline policy demonstration; no model or provider claims")
    offline.add_argument("--fixture", default="fixtures/outbreak-level-5.json")
    live = commands.add_parser("run", help="Pull real ticks and run the CrewAI hierarchy")
    live.add_argument("--scenario", choices=["outbreak-level-5", "control-tick"], default="outbreak-level-5")
    live.add_argument("--ticks", type=int, choices=range(1, 21), default=1)
    live.add_argument("--submit", action="store_true", help="Explicitly enable scored submissions")
    live.add_argument("--discord", action="store_true", help="Enable incident posting through the configured Discord transport")
    live.add_argument("--discord-transport", choices=["auto", "one", "webhook"], default=os.getenv("DISCORD_TRANSPORT", "auto"))
    notifications = commands.add_parser("notify", help="Post saved accepted decisions without pulling or submitting scenario ticks")
    notifications.add_argument("--tick-id", type=int)
    notifications.add_argument("--discord-transport", choices=["auto", "one", "webhook"], default=os.getenv("DISCORD_TRANSPORT", "auto"))
    notifications.set_defaults(discord=True)
    commands.add_parser("scoreboard", help="Read-only live scoreboard")
    resolve = commands.add_parser("resolve", help="Resolve a terminal incident or ambiguous Discord delivery")
    resolve.add_argument("--tick-id", type=int, required=True)
    resolve.add_argument("--resolution", choices=["retry", "abandon", "delivered"], required=True)
    resolve.add_argument("--message-id")
    resolve.add_argument("--confirm-no-discord-message", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            for data in json.loads(Path(args.fixture).read_text(encoding="utf-8"))["ticks"]:
                tick = Tick.model_validate(data)
                emit("OFFLINE_POLICY_DEMO", tick_id=tick.tick_id, decision=safety_floor(tick).model_dump(),
                     live_services_used=False, scored=False)
        elif args.command == "scoreboard":
            from .connections import MCPConnection
            c = MCPConnection(os.environ["SCENARIO_MCP"])
            try:
                emit("scoreboard", data=c.call("hack26-get-scoreboard"))
            finally:
                c.close()
        elif args.command == "notify":
            from .notifier import notify
            team = team_handle(args.team)
            Path(args.db).parent.mkdir(parents=True, exist_ok=True)
            with process_lock(args.db + ".lock"):
                store = Store(args.db)
                rows = [store.get(team, args.tick_id)] if args.tick_id is not None else store.pending(team)
                if args.tick_id is not None and rows == [None]:
                    raise ValueError("Unknown tick")
                unsent = False
                for row in rows:
                    if not row["judgment"] or not row["decision"] or row["status"] != "active":
                        emit("escalation_skipped", tick_id=row["tick_id"], reason="No active accepted decision")
                        continue
                    escalate(Tick.model_validate_json(row["tick"]), json.loads(row["decision"]),
                             json.loads(row["judgment"]), args, team, store, notify)
                    state = store.get(team, row["tick_id"])
                    delivery = json.loads(state["escalation"]) if state["escalation"] else {}
                    unsent |= delivery.get("status") in ("pending", "unknown")
                if unsent:
                    raise RuntimeError("Some notifications remain pending or unverified")
        elif args.command == "resolve":
            team = team_handle(args.team)
            Path(args.db).parent.mkdir(parents=True, exist_ok=True)
            with process_lock(args.db + ".lock"):
                store = Store(args.db)
                row = store.get(team, args.tick_id)
                if not row:
                    raise ValueError("Unknown tick")
                unknown = row["escalation"] and json.loads(row["escalation"]).get("status") == "unknown"
                if args.resolution == "retry":
                    if unknown and not args.confirm_no_discord_message:
                        raise ValueError("Check Discord, then confirm no message before a retry")
                    store.status(team, args.tick_id, "active")
                    if row["escalation"]:
                        store.update(team, args.tick_id, "escalation", {"status": "pending"})
                elif args.resolution == "abandon":
                    store.status(team, args.tick_id, "abandoned")
                else:
                    if not args.message_id or not args.message_id.isdigit() or not row["judgment"]:
                        raise ValueError("Delivered resolution requires a message ID and accepted judgment")
                    store.update(team, args.tick_id, "escalation", {"status": "user_confirmed", "message_id": args.message_id})
                emit("resolved", tick_id=args.tick_id, resolution=args.resolution)
        else:
            Path(args.db).parent.mkdir(parents=True, exist_ok=True)
            with process_lock(args.db + ".lock"):
                run(args)
    except Exception as exc:
        # Exceptions from HTTP libraries can include authenticated URLs. Print types only.
        emit("error", type=type(exc).__name__, message="Operation stopped. Inspect local state before retrying.")
        sys.exit(1)


if __name__ == "__main__":
    main()
