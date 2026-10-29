"""Synthetic ticks for practice: random sectors that cover every branch of the containment rule.

The venue feed only has three ticks (142, 143, 900). These are generated locally, never submitted, and
use tick ids from 800000 up so they cannot be mistaken for venue ticks. The expected action comes from
contracts.safety_floor, the published fixed rule that every venue judgment so far has matched.
"""

import random

from .contracts import Tick, safety_floor

SECTORS = ["catering_coffee_bar", "main_stage", "entrance", "server_room", "lobby", "workshop_bay",
           "rooftop_terrace", "green_room", "loading_dock", "quiet_pod"]
THREATS = ["level-5-phantom", "level-3-specter", "poltergeist", "wraith", "shade", "banshee"]
HARMLESS = ["human-crowd", "residual-echo"]
AT_RISK = ["{n} pinned", "{n} trapped", "{n} injured", "{n} missing", "{n} stranded"]
SAFE = ["clear", "0 trapped", "no injured", "evacuated"]


def _sensors(rng, anomalous):
    """Anomalous sectors trip 2-3 indicators; normal ones trip at most one. Values sometimes sit on the
    thresholds (EMF 60, 5 C, 10 motion events), where a careless reading flips the answer."""
    if anomalous:
        emf, temp, motion = rng.randint(60, 160), rng.randint(-12, 5), rng.randint(10, 40)
        drop = rng.choice([None, None, "emf", "temp", "motion"])  # sometimes exactly two indicators
    else:
        emf, temp, motion = rng.randint(0, 59), rng.randint(6, 24), rng.randint(0, 9)
        drop = None
        keep = rng.choice([None, None, "motion", "emf", "temp"])  # at most one indicator: still normal
        if keep == "motion":
            motion = rng.randint(10, 60)  # crowd activity
        elif keep == "emf":
            emf = rng.randint(60, 90)
        elif keep == "temp":
            temp = rng.randint(-3, 5)
    if drop == "emf":
        emf = rng.randint(5, 59)
    elif drop == "temp":
        temp = rng.randint(6, 22)
    elif drop == "motion":
        motion = rng.randint(0, 9)
    if rng.random() < 0.25:  # edge values
        emf = 60 if emf >= 60 else 59
    return {"emf_milligauss": emf, "temp_c": temp, "motion_events": motion}


def _sector(rng, name, profile):
    anomalous = profile in ("contain", "evacuate", "anomaly_only")
    sighting = None
    if profile in ("contain", "evacuate"):
        credible = rng.random() < 0.7
        sighting = ({"entity_class": rng.choice(THREATS), "confidence": round(rng.uniform(0.65, 0.98), 2),
                     "witness_count": rng.randint(2, 7)} if credible else None)
        breached = not credible or rng.random() < 0.5  # a threat needs a credible sighting or a breach
    elif profile == "false_alarm":
        sighting = {"entity_class": rng.choice(HARMLESS), "confidence": round(rng.uniform(0.3, 0.99), 2),
                    "witness_count": rng.randint(0, 3)}
        breached = False
    elif profile == "rumor":  # a scary entity but too weak to count: low confidence or a single witness
        sighting = {"entity_class": rng.choice(THREATS), "confidence": round(rng.uniform(0.2, 0.64), 2),
                    "witness_count": rng.randint(0, 1)}
        breached = False
    else:  # quiet, anomaly_only, trapped_no_threat
        breached = False
    at_risk = profile == "evacuate" or profile == "trapped_no_threat"
    status = rng.choice(AT_RISK).format(n=rng.randint(1, 6)) if at_risk else rng.choice(SAFE)
    return {"sector": name, "sensors": _sensors(rng, anomalous), "sighting": sighting,
            "civilian_status": status, "containment_state": "breached" if breached else "holding"}


def random_tick(seed=None):
    """A valid random Tick plus the action the fixed rule expects."""
    rng = random.Random(seed)
    names = rng.sample(SECTORS, rng.randint(1, 3))
    profiles = rng.choices(["quiet", "false_alarm", "rumor", "anomaly_only", "trapped_no_threat", "contain", "evacuate"],
                           weights=[3, 3, 2, 2, 1, 2, 3], k=len(names))
    tick = Tick.model_validate({
        "tick_id": 800000 + rng.randint(0, 199999),
        "facility_id": "aws-builder-loft-sf",
        "sector_reports": [_sector(rng, n, p) for n, p in zip(names, profiles)],
        "swarm_budget": {"max_active_agents": 8, "api_credits_remaining": 45000},
        "escalation_channel": {"app": "discord", "channel": "#loft-incident", "via": "one-mcp"},
    })
    return tick, safety_floor(tick).action
