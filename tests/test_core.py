import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from agentgeist import report
from agentgeist.budget import BudgetExceeded, Ledger
from agentgeist.connections import Partners, decode_tool, describe_error, one_env, with_retries
from agentgeist.contracts import Decision, Tick, civilian_risk, safety_floor, team_handle
from agentgeist.store import Store, process_lock
from agentgeist.swarm import SPECIALISTS, chat_model, sensor_intel
from agentgeist.trace import Trace, describe, record_outcome


def test_clients_are_built_once_per_process(monkeypatch):
    # Each new HTTPS client costs ~0.3 s on Windows; rebuilding them per tick cost ~0.8 s.
    monkeypatch.setenv("PARASAIL_API_KEY", "k")
    first = Partners(Ledger(8, 1000))
    first.close()
    assert Partners(Ledger(8, 1000)).client is first.client and not first.client.is_closed
    assert chat_model("some-model") is chat_model("some-model")


def fixture(name="outbreak-level-5"):
    return json.loads(Path(f"fixtures/{name}.json").read_text())["ticks"]


def test_process_lock_is_exclusive_and_released(tmp_path):
    path = str(tmp_path / "db.lock")
    with process_lock(path):
        with pytest.raises(OSError):
            with process_lock(path): pass
    with process_lock(path): pass


def test_one_env_passes_key_but_not_other_secrets(monkeypatch):
    monkeypatch.setenv("ONE_API_KEY", "sk_test_one")
    monkeypatch.setenv("PARASAIL_API_KEY", "leak")
    env = one_env()
    assert env["ONE_SECRET"] == "sk_test_one" and env["HOME"]
    assert "leak" not in env.values()


def test_trace_writes_only_when_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTGEIST_TRACE", raising=False)
    Trace(1).event("tool", role="r")
    path = tmp_path / "t" / "trace.jsonl"
    monkeypatch.setenv("AGENTGEIST_TRACE", str(path))
    Trace(142).event("model_call", role="Sensor analyst", answer="ok")
    (event,) = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert event["tick_id"] == 142 and event["kind"] == "model_call" and event["answer"] == "ok"


def tick_folder(tmp_path, monkeypatch, output="a: anomalous"):
    monkeypatch.setenv("AGENTGEIST_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("AGENTGEIST_TRACE", raising=False)
    tick = Tick.model_validate(fixture()[0])
    trace = Trace(tick.tick_id)
    trace.record_call([{"role": "system", "content": "You are Sensor analyst. Rows."},
                       {"role": "user", "content": 'Current Task: Data: [["a",1]]\n\nRules'}],
                      f"Thought: high EMF\nFinal Answer: {output}", 1.2, 900, "m")
    result = {"decision": {"action": "evacuate", "rationale": "r"}, "proposed": {"action": "evacuate", "rationale": "r"},
              "supervisor_override": False, "model_decision_valid": True, "credits_reserved": 900, "peak_active_agents": 4}
    return Path(trace.save_run(tick, result, {"Sensor analyst": [{"sector": "a"}]}))


def test_tick_log_folder_has_a_file_per_agent(tmp_path, monkeypatch):
    folder = tick_folder(tmp_path, monkeypatch)
    record_outcome(folder, team="agentgeist-test", judgment={"verdict": "correct", "score": 1})
    assert sorted(p.name for p in folder.iterdir()) == ["0_tick.json", "1_sensor_analyst.md", "5_outcome.json",
                                                         "run.json", "summary.md"]
    page = (folder / "1_sensor_analyst.md").read_text(encoding="utf-8")
    assert "high EMF" in page and "a: anomalous" in page and '"sector": "a"' in page
    assert "Verdict: **correct**" in (folder / "summary.md").read_text(encoding="utf-8")


def one_gmail(monkeypatch, send_status=200):
    sent = []

    def handler(request):
        if request.url.path == "/v1/vault/connections":
            return httpx.Response(200, json={"rows": [{"platform": "gmail", "key": "live::gmail::default::abc"}]})
        sent.append(request)
        if send_status != 200:
            return httpx.Response(send_status, json={"message": "nope"})
        return httpx.Response(200, json={"email": {"messageId": "m1", "sent": True}})
    real = httpx.Client
    monkeypatch.setattr(report.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setenv("ONE_API_KEY", "sk_test_one")
    monkeypatch.delenv("ONE_GMAIL_CONNECTION_KEY", raising=False)
    return sent


def test_email_report_off_without_address(tmp_path, monkeypatch):
    sent = one_gmail(monkeypatch)
    monkeypatch.delenv("AGENTGEIST_REPORT_EMAIL", raising=False)
    assert report.email_report(tick_folder(tmp_path, monkeypatch)) is None and sent == []


def test_email_report_sends_explanation_and_logs_through_one(tmp_path, monkeypatch):
    sent = one_gmail(monkeypatch)
    monkeypatch.setenv("AGENTGEIST_REPORT_EMAIL", "me@example.com")
    folder = tick_folder(tmp_path, monkeypatch, output="<script>x</script> bar: anomalous")
    record_outcome(folder, team="agentgeist-test", judgment={"verdict": "correct", "score": 1, "false_alarm": False})
    assert report.email_report(folder)["status"] == "sent"
    (request,) = sent
    payload = json.loads(request.content)
    assert request.url.path == "/v1/passthrough/v1/gmail/send-email"
    assert request.headers["x-one-action-id"] == report.SEND_ACTION
    assert payload["to"] == "me@example.com" and payload["isHtml"] and payload["connectionKey"].startswith("live::gmail")
    assert payload["subject"] == "Agentgeist (agentgeist-test) tick 142: EVACUATE, venue: correct 1"
    assert "judged it correct" in payload["body"] and "1_sensor_analyst.md" in payload["body"]
    assert "<script>" not in payload["body"] and "&lt;script&gt;" in payload["body"]
    assert "Email report: sent" in (folder / "summary.md").read_text(encoding="utf-8")


def test_email_report_failure_is_recorded_not_raised_and_not_retried(tmp_path, monkeypatch):
    sent = one_gmail(monkeypatch, send_status=503)
    monkeypatch.setenv("AGENTGEIST_REPORT_EMAIL", "me@example.com")
    folder = tick_folder(tmp_path, monkeypatch)
    result = report.email_report(folder)
    assert result["status"] == "failed" and result["error"]["status"] == 503 and len(sent) == 1
    assert json.loads((folder / "6_email.json").read_text())["error"]["type"] == "HTTPStatusError"
    assert "Email report: failed" in (folder / "summary.md").read_text(encoding="utf-8")


def test_tick_log_disabled_with_empty_dir(monkeypatch):
    monkeypatch.setenv("AGENTGEIST_LOG_DIR", "")
    assert Trace(1).save_run(Tick.model_validate(fixture()[0])) is None


class Upstream(Exception):
    def __init__(self, status, retry_after=None):
        self.status_code = status
        self.response = type("R", (), {"status_code": status,
                                       "headers": {"retry-after": retry_after} if retry_after else {}})()


def test_with_retries_recovers_from_rate_limit():
    attempts, waits, seen = [], [], []

    def call():
        attempts.append(1)
        if len(attempts) < 3:
            raise Upstream(429, retry_after="30" if len(attempts) == 2 else None)
        return "ok"
    assert with_retries(call, on_retry=lambda *a: seen.append(a), sleep=waits.append) == "ok"
    assert waits == [0.5, 4.0]  # default backoff, then Retry-After capped
    assert [s[:2] for s in seen] == [(429, 1), (429, 2)]


@pytest.mark.parametrize("error", [Upstream(400), Upstream(401), TimeoutError("slow")])
def test_with_retries_does_not_retry_permanent_errors_or_timeouts(error):
    waits = []
    with pytest.raises(type(error)):
        with_retries(lambda: (_ for _ in ()).throw(error), sleep=waits.append)
    assert waits == []


def test_with_retries_gives_up_after_two_retries():
    waits = []
    with pytest.raises(Upstream):
        with_retries(lambda: (_ for _ in ()).throw(Upstream(503)), sleep=waits.append)
    assert waits == [0.5, 1.5]


def test_partner_http_retries_429_then_succeeds(monkeypatch):
    responses = iter([httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(200, json={"data": [
        {"index": 0, "embedding": [1.0]}]})])
    monkeypatch.setenv("PARASAIL_API_KEY", "k")
    partners = Partners(Ledger(8, 10_000))
    partners.client = httpx.Client(transport=httpx.MockTransport(lambda request: next(responses)))
    assert partners.embeddings(["x"]) == [[1.0]]


def http_error(status, url, body):
    request = httpx.Request("POST", url)
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError(f"{status} for url {url}", request=request, response=response)


def test_describe_error_gives_status_service_reason_but_never_the_url():
    secret = "https://discord.com/api/webhooks/1/SECRET-TOKEN"
    info = describe_error(http_error(404, secret, {"message": "Unknown Webhook"}))
    assert info == {"type": "HTTPStatusError", "status": 404, "service": "Discord"}
    assert "SECRET" not in json.dumps(info)
    overloaded = http_error(429, "https://api.parasail.io/v1/embeddings",
                            {"error": {"message": "The engine is currently overloaded. Please try again later."}})
    assert describe_error(overloaded) == {"type": "HTTPStatusError", "status": 429, "service": "Parasail",
                                          "reason": "The engine is currently overloaded. Please try again later."}


def test_describe_error_finds_the_status_inside_a_wrapped_error():
    try:
        try:
            raise http_error(503, "https://ydc-index.io/v1/search", {})
        except httpx.HTTPStatusError as inner:
            raise RuntimeError("Missing specialist tool evidence") from inner
    except RuntimeError as outer:
        info = describe_error(outer)
    assert info["type"] == "RuntimeError" and info["status"] == 503 and info["service"] == "You.com"
    assert describe(info) == "`RuntimeError` HTTP 503 from You.com"


class DownEmbeddings:
    def __init__(self, error, trace=None):
        self.error, self.trace = error, trace

    def embeddings(self, texts):
        raise self.error


def test_sensor_intel_continues_without_embeddings(monkeypatch):
    monkeypatch.delenv("AGENTGEIST_TRACE", raising=False)
    trace = Trace(142)
    sectors = [{"sector": "bar", "sensors": {"emf_milligauss": 98, "temp_c": -4, "motion_events": 17}}]
    overloaded = http_error(429, "https://api.parasail.io/v1/embeddings", {"error": {"message": "overloaded"}})
    (row,) = sensor_intel(sectors, DownEmbeddings(overloaded, trace))
    assert row["physical_indicators"] == 3 and row["anomaly_similarity"] is None and row["normal_similarity"] is None
    assert SPECIALISTS[0][3]([row])[0][-2:] == [None, None]
    assert trace.degraded[0]["service"] == "Parasail embeddings" and trace.degraded[0]["error"]["status"] == 429


def test_sensor_intel_still_stops_on_budget():
    sectors = [{"sector": "bar", "sensors": {"emf_milligauss": 1, "temp_c": 20, "motion_events": 0}}]
    with pytest.raises(BudgetExceeded):
        sensor_intel(sectors, DownEmbeddings(BudgetExceeded("over")))


def test_team_normalization():
    assert team_handle(" Agentgeist ") == "agentgeist"


@pytest.mark.parametrize("team", ["x", "a_b", "bad/name", "a" * 33])
def test_bad_team(team):
    with pytest.raises(ValueError): team_handle(team)


def test_every_sector_is_used():
    d = fixture()[0]
    d["sector_reports"].reverse()
    assert safety_floor(Tick.model_validate(d)).action == "evacuate"


@pytest.mark.parametrize("data", fixture())
def test_outbreak(data):
    assert safety_floor(Tick.model_validate(data)).action == "evacuate"


def test_crowd_motion_is_not_a_threat():
    assert safety_floor(Tick.model_validate(fixture("control-tick")[0])).action == "observe"


def test_contain_without_civilians():
    data = fixture()[0]
    data["sector_reports"][0]["civilian_status"] = "clear"
    assert safety_floor(Tick.model_validate(data)).action == "contain"


def test_fixed_field_names():
    data = fixture()[0]
    data["sectors"] = data.pop("sector_reports")
    with pytest.raises(ValidationError): Tick.model_validate(data)


def test_duplicate_sectors_rejected():
    data = fixture()[0]
    data["sector_reports"] *= 2
    with pytest.raises(ValidationError): Tick.model_validate(data)


def test_agent_limit_and_release():
    ledger = Ledger(3, 10)
    with pytest.raises(BudgetExceeded):
        with ledger.agents(4): pass
    with pytest.raises(RuntimeError):
        with ledger.agents(3): raise RuntimeError()
    assert ledger.active == 0 and ledger.peak == 3


def test_credit_reservation():
    ledger = Ledger(4, 10)
    ledger.reserve(10)
    with pytest.raises(BudgetExceeded): ledger.reserve(1)
    assert ledger.remaining == 0


@pytest.mark.parametrize("status,expected", [("4 pinned", True), ("clear", False), ("0 pinned", False), ("no injured", False)])
def test_civilian_status(status, expected):
    assert civilian_risk(status) is expected


def test_decision_enum():
    with pytest.raises(ValidationError): Decision(action="attack", rationale="unsafe")


def test_tool_error_is_not_success():
    with pytest.raises(RuntimeError): decode_tool({"isError": True, "content": []})
    assert decode_tool({"content": [{"type": "text", "text": '{"score":1}'}]}) == {"score": 1}


def test_durable_outbox(tmp_path):
    path = str(tmp_path / "state.db")
    s = Store(path)
    tick = Tick.model_validate(fixture()[0])
    s.save_tick("agentgeist", tick)
    s.save_tick("agentgeist", tick)
    s.update("agentgeist", tick.tick_id, "decision", {"action": "evacuate"})
    assert len(Store(path).pending("agentgeist")) == 1
    assert not s.pending("other-team")
    s.update("agentgeist", tick.tick_id, "judgment", {"score": 1})
    s.update("agentgeist", tick.tick_id, "escalation", {"status": "sent"})
    assert not s.pending("agentgeist")


def test_specialists_share_evidence_reference():
    from agentgeist.swarm import ScopedTool
    evidence = {}
    tool = ScopedTool(name="triage", description="test", callback=lambda: {"clear": True}, evidence=evidence)
    tool._run()
    assert evidence == {"triage": {"clear": True}}


def test_discord_connection_and_receipt_guards():
    from agentgeist.notifier import NoDiscordConnection, message_receipt, select_connection
    with pytest.raises(NoDiscordConnection): select_connection({"connections": [{"platform": "github"}]})
    with pytest.raises(RuntimeError): message_receipt({"id": "message", "channel_id": "wrong"})
    assert message_receipt({"data": {"id": "123", "channel_id": "1564842098740695100"}})["status"] == "sent"


def test_credits_survive_restart(tmp_path):
    from agentgeist.cli import tick_ledger
    path = str(tmp_path / "state.db")
    store = Store(path)
    tick = Tick.model_validate(fixture()[0])
    store.save_tick("agentgeist", tick)
    tick_ledger(tick, "agentgeist", store).reserve(500)
    restarted = tick_ledger(tick, "agentgeist", Store(path))
    assert restarted.remaining == 44500


def test_team_credit_ceiling_across_ticks(tmp_path):
    from agentgeist.cli import tick_ledger
    store = Store(str(tmp_path / "state.db"))
    first, second = [Tick.model_validate(t) for t in fixture()]
    store.save_tick("agentgeist", first)
    tick_ledger(first, "agentgeist", store).reserve(30000)
    store.save_tick("agentgeist", second)
    ledger = tick_ledger(second, "agentgeist", store)
    assert ledger.remaining == 15000
    with pytest.raises(BudgetExceeded): ledger.reserve(15001)


def test_terminal_and_raw_storage(tmp_path):
    store = Store(str(tmp_path / "state.db"))
    assert store.save_raw("agentgeist", {"unexpected": True}) == 1
    tick = Tick.model_validate(fixture()[0])
    store.save_tick("agentgeist", tick)
    store.status("agentgeist", tick.tick_id, "failed")
    assert not store.pending("agentgeist")
    assert store.get("agentgeist", tick.tick_id)["status"] == "failed"


def test_manager_cannot_redelegate_private_context():
    from agentgeist.swarm import ScopedCrew
    sentinel = object()
    class Fake:
        manager_agent = sentinel
    assert ScopedCrew._prepare_tools(Fake(), sentinel, None, ["secret-context-tool"]) == []
