import json
import threading

import httpx
import pytest

from agentgeist import dashboard


def test_read_run_stays_inside_the_log_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTGEIST_LOG_DIR", str(tmp_path / "logs"))
    tick = tmp_path / "logs" / "2026_tick-1"
    tick.mkdir(parents=True)
    (tick / "summary.md").write_text("# ok", encoding="utf-8")
    (tmp_path / "secret.json").write_text("{}", encoding="utf-8")
    assert dashboard.read_run("2026_tick-1") == {"summary.md": "# ok"}
    for escape in ("..", "../secret.json", "2026_tick-1/../..", str(tmp_path)):
        with pytest.raises(FileNotFoundError):
            dashboard.read_run(escape)


def test_read_events_returns_only_complete_lines(tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("AGENTGEIST_TRACE", str(trace))
    trace.write_text(json.dumps({"kind": "tool"}) + "\n" + '{"kind": "mod', encoding="utf-8")
    assert dashboard.read_events(-1)["events"] == []  # the page starts at the end
    first = dashboard.read_events(0)
    assert first["events"] == [{"kind": "tool"}]
    with open(trace, "a", encoding="utf-8") as f:
        f.write('el_call"}\n')
    assert dashboard.read_events(first["offset"])["events"] == [{"kind": "model_call"}]


@pytest.fixture
def server():
    srv = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.mark.parametrize("body", [
    {"kind": "live", "team": "agentgeist-test9", "scenario": "outbreak-level-5"},                      # no confirmation
    {"kind": "live", "team": "Bad Team!", "scenario": "outbreak-level-5", "confirm": "LIVE"},          # invalid handle
    {"kind": "fixture", "scenario": "../../etc"},                                                      # unknown scenario
])
def test_start_refuses_unconfirmed_or_invalid_runs(server, body):
    response = httpx.post(server + "/api/start", json=body)
    assert response.status_code == 400 and dashboard.RUNNER.current is None
