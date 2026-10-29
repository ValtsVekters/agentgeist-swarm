import json
import threading

import httpx
import pytest

from agentgeist import dashboard, notifier


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


def test_observe_never_posts_to_discord(monkeypatch):
    monkeypatch.setattr(notifier, "notify", lambda *a, **k: pytest.fail("observe must not post"))
    assert dashboard.post_discord("agentgeist", None, {"decision": {"action": "observe"}}, None)["status"] != "posted"


def test_discord_posts_are_marked_practice_under_the_real_team(monkeypatch):
    sent = {}

    def fake_notify(team, tick, result, judgment, ledger, transport=None):
        sent.update(team=team, judgment=judgment, transport=transport)
        return {"status": "sent", "message_url": "https://discord.com/channels/1/2/3"}
    monkeypatch.setattr(notifier, "notify", fake_notify)
    monkeypatch.setenv("DISCORD_TRANSPORT", "webhook")
    out = dashboard.post_discord("agentgeist", None, {"decision": {"action": "evacuate"}}, None)
    assert out == {"status": "posted", "url": "https://discord.com/channels/1/2/3"}
    assert sent["team"] == "agentgeist" and "practice" in sent["judgment"]["score"] and sent["transport"] == "webhook"


def test_discord_failure_is_reported_not_raised(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("webhook down")
    monkeypatch.setattr(notifier, "notify", broken)
    assert dashboard.post_discord("agentgeist", None, {"decision": {"action": "contain"}}, None)["status"] == "failed"


@pytest.fixture
def server():
    srv = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.mark.parametrize("body", [
    {"kind": "saved", "tick": "999"}, {"kind": "saved", "tick": "../../etc"},   # only 142, 143, 900
    {"kind": "random", "count": 0}, {"kind": "random", "count": 11},           # 1..10 only
    {"kind": "live", "team": "agentgeist"},                                    # live runs are not offered here
])
def test_start_refuses_invalid_runs(server, body):
    response = httpx.post(server + "/api/start", json=body, headers={"X-Agentgeist-Token": dashboard.TOKEN})
    assert response.status_code == 400 and dashboard.RUNNER.current is None


VALID = {"kind": "saved", "tick": "142", "discord": True}


@pytest.mark.parametrize("headers", [
    {},                                                                       # no token: another site cannot know it
    {"X-Agentgeist-Token": "guess"},                                          # wrong token
    {"X-Agentgeist-Token": dashboard.TOKEN, "Origin": "https://evil.example"},  # cross-site page
    {"X-Agentgeist-Token": dashboard.TOKEN, "Host": "evil.example:8765"},     # DNS rebinding
])
def test_start_refuses_cross_site_and_rebinding_requests(server, headers):
    response = httpx.post(server + "/api/start", json=VALID, headers=headers)
    assert response.status_code == 403 and dashboard.RUNNER.current is None


def test_start_refuses_form_posts_even_with_token(server):
    # A plain form or text/plain fetch skips the browser's CORS preflight, so it must be refused.
    response = httpx.post(server + "/api/start", content=json.dumps(VALID),
                          headers={"Content-Type": "text/plain", "X-Agentgeist-Token": dashboard.TOKEN})
    assert response.status_code == 403


def test_reads_refuse_foreign_host_and_page_carries_token(server):
    assert httpx.get(server + "/api/status", headers={"Host": "evil.example:8765"}).status_code == 403
    page = httpx.get(server + "/")
    assert page.status_code == 200 and dashboard.TOKEN in page.text and page.headers["X-Frame-Options"] == "DENY"
