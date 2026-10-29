import threading
import time

import pytest

from agentgeist import connections
from agentgeist.budget import Ledger
from agentgeist.connections import Partners
from agentgeist.trace import Trace


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(connections, "_LORE_CACHE", {})
    monkeypatch.setattr(connections, "_LORE_PENDING", {})
    monkeypatch.delenv("AGENTGEIST_TRACE", raising=False)


def partners(monkeypatch, lookup):
    p = Partners(Ledger(8, 10_000), Trace(1))
    monkeypatch.setattr(p, "_lookup", lookup)
    return p


def test_no_sightings_means_no_lookup(monkeypatch):
    p = partners(monkeypatch, lambda key: pytest.fail("no lookup without sightings"))
    assert p.research([])["sources"] == []


def test_repeat_entities_come_from_the_cache(monkeypatch):
    calls = []
    p = partners(monkeypatch, lambda key: calls.append(key) or {"sources": [{"title": "t"}], "warning": "w"})
    p.research(["wraith"])
    time.sleep(0.05)  # let the done-callback store the result
    second = p.research(["wraith", "wraith"])
    assert second["cached"] and len(calls) == 1 and p.ledger.spent == 1000  # a cache hit reserves nothing


def test_slow_lookup_continues_without_lore_and_fills_the_cache_later(monkeypatch):
    monkeypatch.setattr(connections, "LORE_BUDGET_S", 0.05)
    release = threading.Event()

    def slow(key):
        release.wait(2)
        return {"sources": [{"title": "late"}], "warning": "w"}
    p = partners(monkeypatch, slow)
    first = p.research(["banshee"])
    assert first["unavailable"] and first["sources"] == []
    assert p.trace.degraded[0]["service"] == "You.com lore" and p.trace.degraded[0]["error"]["type"] == "Timeout"
    release.set()
    time.sleep(0.1)
    assert p.research(["banshee"])["sources"] == [{"title": "late"}]


def test_failed_lookup_continues_without_lore(monkeypatch):
    def broken(key):
        raise RuntimeError("search down")
    p = partners(monkeypatch, broken)
    assert p.research(["shade"])["unavailable"]
    assert p.trace.degraded[0]["error"]["type"] == "RuntimeError"
