import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentgeist.budget import BudgetExceeded, Ledger
from agentgeist.connections import decode_tool, one_env
from agentgeist.contracts import Decision, Tick, civilian_risk, safety_floor, team_handle
from agentgeist.store import Store, process_lock
from agentgeist.trace import Trace, record_outcome


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


def test_tick_log_folder_has_a_file_per_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTGEIST_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("AGENTGEIST_TRACE", raising=False)
    tick = Tick.model_validate(fixture()[0])
    trace = Trace(tick.tick_id)
    trace.record_call([{"role": "system", "content": "You are Sensor analyst. Rows."},
                       {"role": "user", "content": 'Current Task: Data: [["a",1]]\n\nRules'}],
                      "Thought: high EMF\nFinal Answer: a: anomalous", 1.2, 900, "m")
    result = {"decision": {"action": "evacuate", "rationale": "r"}, "proposed": {"action": "evacuate", "rationale": "r"},
              "supervisor_override": False, "model_decision_valid": True, "credits_reserved": 900, "peak_active_agents": 4}
    folder = Path(trace.save_run(tick, result, {"Sensor analyst": [{"sector": "a"}]}))
    record_outcome(folder, team="agentgeist-test", judgment={"verdict": "correct", "score": 1})
    assert sorted(p.name for p in folder.iterdir()) == ["0_tick.json", "1_sensor_analyst.md", "5_outcome.json", "summary.md"]
    page = (folder / "1_sensor_analyst.md").read_text(encoding="utf-8")
    assert "high EMF" in page and "a: anomalous" in page and '"sector": "a"' in page
    assert "Verdict: **correct**" in (folder / "summary.md").read_text(encoding="utf-8")


def test_tick_log_disabled_with_empty_dir(monkeypatch):
    monkeypatch.setenv("AGENTGEIST_LOG_DIR", "")
    assert Trace(1).save_run(Tick.model_validate(fixture()[0])) is None


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
