import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class Sensors(Contract):
    emf_milligauss: float
    temp_c: float
    motion_events: int = Field(ge=0)


class Sighting(Contract):
    entity_class: str
    confidence: float = Field(ge=0, le=1)
    witness_count: int = Field(ge=0)


class Sector(Contract):
    sector: str = Field(min_length=1)
    sensors: Sensors
    sighting: Sighting | None
    civilian_status: str
    containment_state: str


class Budget(Contract):
    max_active_agents: StrictInt = Field(ge=0)
    api_credits_remaining: StrictInt = Field(ge=0)


class Channel(Contract):
    app: Literal["discord"]
    channel: Literal["#loft-incident"]
    via: Literal["one-mcp"]


class Tick(Contract):
    tick_id: StrictInt = Field(gt=0)
    facility_id: str
    sector_reports: list[Sector] = Field(min_length=1)
    swarm_budget: Budget
    escalation_channel: Channel

    @field_validator("sector_reports")
    @classmethod
    def unique_sectors(cls, value):
        if len({s.sector for s in value}) != len(value):
            raise ValueError("Duplicate sectors")
        return value


class Decision(BaseModel):
    action: Literal["contain", "evacuate", "observe"]
    rationale: str = Field(min_length=1, max_length=1500)


def team_handle(value: str) -> str:
    value = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,31}", value):
        raise ValueError("Team handle must contain 2–32 lowercase letters, digits, or hyphens")
    return value


def civilian_risk(status: str) -> bool:
    status = status.lower().strip()
    return bool(re.search(r"\b(pinned|trapped|injured|missing|stranded)\b", status)) and not bool(
        re.match(r"^(?:0|no)\s", status)
    )


def physical_anomaly(sector: Sector) -> bool:
    s = sector.sensors
    return sum((s.emf_milligauss >= 60, s.temp_c <= 5, s.motion_events >= 10)) >= 2


def credible_sighting(sector: Sector) -> bool:
    s = sector.sighting
    return bool(s and s.confidence >= .65 and s.witness_count >= 2
                and s.entity_class not in ("human-crowd", "residual-echo"))


def breached(sector: Sector) -> bool:
    return sector.containment_state == "breached"


def safety_floor(tick: Tick) -> Decision:
    """Transparent simulation policy. Research cannot override physical danger."""
    threats = []
    for s in tick.sector_reports:
        physical, credible = physical_anomaly(s), credible_sighting(s)
        if civilian_risk(s.civilian_status) and (physical or credible or breached(s)):
            return Decision(action="evacuate", rationale=f"{s.sector}: civilians at risk; isolate the sector and evacuate.")
        if physical and (credible or breached(s)):
            threats.append(s.sector)
    if threats:
        return Decision(action="contain", rationale="Corroborated threat in " + ", ".join(threats))
    return Decision(action="observe", rationale="No corroborated threat or civilian danger; avoid a false alarm.")
