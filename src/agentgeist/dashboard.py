"""Local dashboard: run test or live ticks, watch the agents live, and read every tick's logs.

    uv run agentgeist-ui            # opens http://127.0.0.1:8765
    uv run agentgeist-ui --no-browser --port 8765

Serves only on 127.0.0.1. Test ticks run in this process (warmed up once, so they show the real decision time).
Live ticks run the normal `agentgeist run` CLI as a child process, with its database, lock and safety checks.
The page never receives keys or the webhook URL: it sees tick data, agent prompts and answers, and results.
"""

import argparse
import collections
import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from .contracts import Tick, safety_floor, team_handle
from .scenarios import random_tick

SCENARIOS = ("outbreak-level-5", "control-tick")
LIVE_EVENTS = ("decision", "judgment", "discord_receipt", "escalation_pending", "escalation_skipped", "tick_failed",
               "email_report", "submission_rejected", "receipt_unverified", "error", "scoreboard")


class Runner:
    """One run at a time; the page polls status, live agent events and live CLI output."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = None  # {"kind", "label", "started"}
        self.last = None     # last finished run summary
        self.warm = "warming up"
        self.output = collections.deque(maxlen=300)

    def busy(self):
        return self.current is not None

    def start(self, kind, label, target):
        with self.lock:
            if self.current:
                raise RuntimeError("A run is already in progress")
            self.current = {"kind": kind, "label": label, "started": time.time()}
            self.output.clear()
        threading.Thread(target=self._wrap, args=(target,), daemon=True).start()

    def _wrap(self, target):
        try:
            self.last = {**self.current, **target(), "finished": time.time()}
        except Exception as exc:
            from .connections import describe_error
            self.last = {**self.current, "ok": False, "error": describe_error(exc), "finished": time.time()}
        finally:
            with self.lock:
                self.current = None


RUNNER = Runner()


def without_email(enabled):
    """Child environment and in-process toggle for the email report (one run at a time, so no race)."""
    env = dict(os.environ)
    if not enabled:
        env.pop("AGENTGEIST_REPORT_EMAIL", None)
    return env


def saved_tick(scenario, index):
    data = json.loads(files("agentgeist").joinpath("fixtures", scenario + ".json").read_text(encoding="utf-8"))
    tick = Tick.model_validate(data["ticks"][index])
    return tick, safety_floor(tick).action, None


def random_ticks(count, seed):
    seed = secrets.randbelow(10**6) if seed is None else seed
    return [(*random_tick(seed + i), seed + i) for i in range(count)]


def run_ticks(ticks, email):
    """Run (tick, expected, seed) items through the warmed swarm like `kickoff`: never submitted or posted.
    Each result compares the commander's own proposal with the fixed rule the venue referee uses."""
    from .budget import Ledger
    from .connections import describe_error
    from .report import email_report
    from .swarm import run_swarm, warm_up
    warm_up()
    saved = os.environ.get("AGENTGEIST_REPORT_EMAIL")
    if not email:
        os.environ.pop("AGENTGEIST_REPORT_EMAIL", None)
    results = []
    try:
        for tick, expected, seed in ticks:
            row = {"tick": tick.tick_id, "seed": seed, "expected": expected, "sectors": len(tick.sector_reports)}
            try:
                result = run_swarm(tick, Ledger(tick.swarm_budget.max_active_agents, tick.swarm_budget.api_credits_remaining))
                folder = Path(result["log_dir"]) if result.get("log_dir") else None
                run = json.loads((folder / "run.json").read_text(encoding="utf-8")) if folder else {}
                mail = email_report(result.get("log_dir"))
                row.update(proposed=result["proposed"]["action"], action=result["decision"]["action"],
                           override=result["supervisor_override"], seconds=run.get("swarm_seconds"),
                           log=folder.name if folder else None, email=mail["status"] if mail else "off",
                           matched=result["proposed"]["action"] == expected)
            except Exception as exc:
                row.update(error=describe_error(exc), matched=False)
            results.append(row)
    finally:
        if saved is not None:
            os.environ["AGENTGEIST_REPORT_EMAIL"] = saved
    last = results[-1]
    return {"ok": "error" not in last, "action": last.get("action"), "seconds": last.get("seconds"),
            "log": last.get("log"), "email": last.get("email"), "results": results,
            "matched": sum(r["matched"] for r in results), "total": len(results)}


def run_live(team, scenario, submit, discord, email):
    cmd = [sys.executable, "-m", "agentgeist.cli", "--team", team, "--db", f"state/{team}.sqlite3",
           "run", "--scenario", scenario, "--ticks", "1"]
    if submit:
        cmd.append("--submit")
    if discord:
        cmd += ["--discord", "--discord-transport", os.getenv("DISCORD_TRANSPORT", "webhook")]
    env = {**without_email(email), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=env)
    summary = {"ok": False}
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue  # CrewAI console noise; the CLI's JSON events carry the results
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") in LIVE_EVENTS:
            RUNNER.output.append(event)
        if event.get("event") == "decision":
            summary.update(ok=True, action=event["decision"]["action"], seconds=event.get("decision_seconds"),
                           log=Path(event["log_dir"]).name if event.get("log_dir") else None)
        elif event.get("event") == "judgment":
            summary["judgment"] = event["judgment"]
        elif event.get("event") == "tick_failed":
            summary.update(ok=False, error={k: v for k, v in event.items() if k not in ("event", "tick_id")})
    summary["exit_code"] = proc.wait()
    return summary


def trace_path():
    return Path(os.getenv("AGENTGEIST_TRACE") or "state/agent-trace.jsonl")


def log_root():
    return Path(os.getenv("AGENTGEIST_LOG_DIR", "logs") or "logs").resolve()


def list_runs(limit=60):
    runs = []
    for run_json in sorted(log_root().rglob("run.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        folder = run_json.parent
        try:
            run = json.loads(run_json.read_text(encoding="utf-8"))
            outcome = folder / "5_outcome.json"
            outcome = json.loads(outcome.read_text(encoding="utf-8")) if outcome.exists() else {}
        except ValueError:
            continue
        judgment = outcome.get("judgment") or {}
        runs.append({"id": str(folder.relative_to(log_root())).replace("\\", "/"), "tick": run.get("tick_id"),
                     "started": run.get("started"), "action": (run.get("decision") or {}).get("action"),
                     "seconds": run.get("swarm_seconds"), "error": run.get("error"), "team": outcome.get("team"),
                     "verdict": judgment.get("verdict"), "score": judgment.get("score"),
                     "override": run.get("supervisor_override")})
    return runs


def read_run(run_id):
    root = log_root()
    folder = (root / run_id).resolve()
    if root not in folder.parents or not folder.is_dir():  # stay inside logs/: no path traversal
        raise FileNotFoundError(run_id)
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix in (".md", ".json")}


def read_events(offset):
    path = trace_path()
    if not path.exists():
        return {"offset": 0, "events": []}
    size = path.stat().st_size
    if offset < 0:
        return {"offset": size, "events": []}  # the page starts at the end: only new activity
    if offset > size:
        offset = 0  # the trace file was replaced
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    complete = chunk.rfind(b"\n") + 1  # never hand back a half-written line
    lines = chunk[:complete].decode("utf-8", errors="replace").splitlines()
    return {"offset": offset + complete, "events": [json.loads(l) for l in lines if l.strip()]}


def scoreboard():
    from .connections import MCPConnection
    connection = MCPConnection(os.environ["SCENARIO_MCP"])
    try:
        return connection.call("hack26-get-scoreboard")
    finally:
        connection.close()


# Per-process secret, embedded in the served page and required to start a run: another site open in the
# browser can reach 127.0.0.1 but cannot read this page, so it cannot learn the token.
TOKEN = secrets.token_urlsafe(32)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the terminal quiet; requests carry no secrets but are noisy

    def origins(self):
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def trusted(self, write=False):
        """Refuse DNS rebinding (foreign Host) on every request, and cross-site requests on writes."""
        if self.headers.get("Host", "") not in self.origins():
            return False
        if not write:
            return True
        origin = self.headers.get("Origin")
        return (self.headers.get("Content-Type", "").startswith("application/json")  # forces a CORS preflight
                and (origin is None or (origin.startswith("http://") and urlparse(origin).netloc in self.origins()))
                and hmac.compare_digest(self.headers.get("X-Agentgeist-Token", ""), TOKEN))

    def send(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")  # no framing: nobody can trick a click on "Run live tick"
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.trusted():
            return self.send(403, {"error": "forbidden host"})
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/":
                page = files("agentgeist").joinpath("dashboard.html").read_bytes()
                self.send(200, page.replace(b"__AGENTGEIST_TOKEN__", TOKEN.encode()), "text/html")
            elif url.path == "/api/status":
                self.send(200, {"running": RUNNER.current, "last": RUNNER.last, "warm": RUNNER.warm,
                                "output": list(RUNNER.output), "model": os.getenv("PARASAIL_CHAT_MODEL"),
                                "email": bool(os.getenv("AGENTGEIST_REPORT_EMAIL")), "team": os.getenv("TEAM_HANDLE"),
                                "discord_transport": os.getenv("DISCORD_TRANSPORT", "auto")})
            elif url.path == "/api/events":
                self.send(200, read_events(int(query.get("offset", ["0"])[0])))
            elif url.path == "/api/runs":
                self.send(200, list_runs())
            elif url.path == "/api/run":
                self.send(200, read_run(query["id"][0]))
            elif url.path == "/api/scoreboard":
                self.send(200, scoreboard())
            else:
                self.send(404, {"error": "not found"})
        except FileNotFoundError:
            self.send(404, {"error": "not found"})
        except Exception as exc:
            from .connections import describe_error
            self.send(500, {"error": describe_error(exc)})

    def do_POST(self):
        # Read the (small, capped) body before answering: replying first makes Windows abort the connection.
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > 65536:
            return self.send(413, {"error": "request too large"})
        raw = self.rfile.read(length)
        if not self.trusted(write=True):
            return self.send(403, {"error": "forbidden: cross-site or missing dashboard token"})
        if urlparse(self.path).path != "/api/start":
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(raw or b"{}")
            kind, email = body.get("kind"), bool(body.get("email"))
            if kind == "random":
                count, seed = int(body.get("count", 1)), body.get("seed")
                if not 1 <= count <= 10:
                    raise ValueError("Run 1 to 10 random ticks at a time")
                seed = None if seed in (None, "") else int(seed)
                ticks = random_ticks(count, seed)
                label = f"Random tick{'s' if count > 1 else ''} (seed {ticks[0][2]}{f'..{ticks[-1][2]}' if count > 1 else ''})"
                RUNNER.start("random", label, lambda: run_ticks(ticks, email))
                return self.send(200, {"started": True})
            scenario = body.get("scenario")
            if scenario not in SCENARIOS:
                raise ValueError("Unknown scenario")
            if kind == "fixture":
                index = int(body.get("index", 0))
                if index not in (0, 1) or (scenario == "control-tick" and index != 0):
                    raise ValueError("Unknown saved tick")
                ticks = [saved_tick(scenario, index)]
                RUNNER.start("fixture", f"Saved tick {ticks[0][0].tick_id}", lambda: run_ticks(ticks, email))
            elif kind == "live":
                if body.get("confirm") != "LIVE":
                    raise ValueError("Live runs need explicit confirmation")
                team = team_handle(body.get("team", ""))
                submit, discord = bool(body.get("submit")), bool(body.get("discord"))
                RUNNER.start("live", f"Live tick: {team} on {scenario}",
                             lambda: run_live(team, scenario, submit, discord, email))
            else:
                raise ValueError("Unknown run kind")
            self.send(200, {"started": True})
        except (ValueError, RuntimeError) as exc:
            self.send(400, {"error": str(exc)})


def main():
    load_dotenv()
    os.environ.setdefault("AGENTGEIST_TRACE", "state/agent-trace.jsonl")  # the live view reads this stream
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    parser = argparse.ArgumentParser(description="Agentgeist dashboard")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def warm():
        from .swarm import warm_up
        try:
            warm_up()
            RUNNER.warm = "ready"
        except Exception as exc:
            RUNNER.warm = f"warm-up failed: {type(exc).__name__}"
    threading.Thread(target=warm, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Agentgeist dashboard on {url} (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
