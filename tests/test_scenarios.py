import collections

from agentgeist.contracts import safety_floor
from agentgeist.scenarios import random_tick


def test_random_ticks_cover_every_action_and_are_reproducible():
    counts = collections.Counter(random_tick(seed)[1] for seed in range(400))
    assert set(counts) == {"observe", "contain", "evacuate"} and min(counts.values()) > 20
    assert random_tick(7)[0] == random_tick(7)[0]


def test_random_ticks_are_valid_and_cannot_be_mistaken_for_venue_ticks():
    for seed in range(200):
        tick, expected = random_tick(seed)
        assert tick.tick_id >= 800000 and 1 <= len(tick.sector_reports) <= 3
        assert expected == safety_floor(tick).action
