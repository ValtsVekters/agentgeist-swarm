"""CrewAI AMP Flow entry point. The default runs real agents on a saved fixture."""

import asyncio
import json
import os
import sys
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from importlib.resources import files
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load tracing switches before importing CrewAI's tracing machinery.
load_dotenv()

from crewai.flow import Flow, start  # noqa: E402

from agentgeist.contracts import Tick, team_handle  # noqa: E402


class RunOptions(BaseModel):
    mode: Literal["fixture", "live"] = "fixture"
    team: str = "agentgeist"
    scenario: Literal["outbreak-level-5", "control-tick"] = "outbreak-level-5"
    submit: bool = False
    discord: bool = False

    @field_validator("team")
    @classmethod
    def normalize_team(cls, value):
        return team_handle(value)


class AgentgeistState(RunOptions):
    result: dict = Field(default_factory=dict)


def execute(options: RunOptions):
    if options.mode == "fixture":
        from agentgeist.budget import Ledger
        from agentgeist.swarm import run_swarm
        data = json.loads(files("agentgeist").joinpath("fixtures", options.scenario + ".json").read_text(encoding="utf-8"))
        tick = Tick.model_validate(data["ticks"][0])
        result = run_swarm(tick, Ledger(tick.swarm_budget.max_active_agents, tick.swarm_budget.api_credits_remaining))
        return {"mode": "fixture", "tick_id": tick.tick_id, "submitted": False, "discord_posted": False, **result}

    if os.getenv("AGENTGEIST_ENABLE_LIVE", "false").lower() != "true":
        raise ValueError("Set AGENTGEIST_ENABLE_LIVE=true only after configuring durable, single-worker state")
    from crewai.events import crewai_event_bus
    from crewai.events.event_listener import event_listener
    from crewai.events.listeners.tracing.utils import is_tracing_enabled_in_context
    from crewai.telemetry.tracing.context import get_trace_session

    # Venue MCP events contain credentials. Never run them inside a traced Flow.
    if (os.getenv("CREWAI_TRACING_ENABLED", "").lower() != "false"
            or is_tracing_enabled_in_context() or get_trace_session() is not None):
        raise ValueError("Live mode requires CREWAI_TRACING_ENABLED=false and no active CrewAI tracing")
    state_directory = os.getenv("AGENTGEIST_STATE_DIR", "")
    if not state_directory or not Path(state_directory).is_absolute():
        raise ValueError("Set AGENTGEIST_STATE_DIR to an absolute durable directory before live mode")

    from agentgeist.cli import run
    from agentgeist.store import Store, process_lock
    directory = Path(state_directory)
    db = str(directory / "agentgeist.sqlite3")
    args = Namespace(team=options.team, db=db, scenario=options.scenario, ticks=1,
                     submit=options.submit, discord=options.discord,
                     discord_transport=os.getenv("DISCORD_TRANSPORT", "webhook"))
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with process_lock(db + ".lock"):
            # MCP console events include authenticated URLs and tool arguments.
            with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
                try:
                    run(args)
                finally:
                    event_listener.formatter.verbose = False
                    if not crewai_event_bus.flush():
                        raise RuntimeError("Live console event shutdown timed out")
            store = Store(db)
            try:
                rows = store.db.execute("SELECT tick_id,judgment,escalation,status FROM incidents WHERE team=? ORDER BY tick_id DESC LIMIT 20",
                                        (options.team,)).fetchall()
                saved = [{"tick_id": row[0], "judgment": json.loads(row[1]) if row[1] else None,
                          "escalation": json.loads(row[2]) if row[2] else None, "status": row[3]} for row in rows]
            finally:
                store.db.close()
    except Exception as error:
        # HTTP exceptions can contain authenticated URLs; do not expose their text.
        raise RuntimeError(f"Live execution failed: {type(error).__name__}") from None
    return {"mode": "live", "team": options.team, "saved_incidents": saved}


class AgentgeistFlow(Flow[AgentgeistState]):
    @start()
    async def respond(self, crewai_trigger_payload: dict | None = None):
        raw = crewai_trigger_payload if crewai_trigger_payload is not None else self.state.model_dump()
        options = RunOptions.model_validate(raw)
        for key, value in options.model_dump().items():
            setattr(self.state, key, value)
        # MCP resolution uses synchronous event-loop helpers. Keep it outside the Flow loop.
        self.state.result = await asyncio.to_thread(execute, options)
        return self.state.result


def kickoff():
    """Standard `crewai run` entry point: real provider calls, no venue side effects."""
    from agentgeist.cli import utf8_console
    utf8_console()
    tracing = os.getenv("CREWAI_TRACING_ENABLED", "").lower()
    flow = AgentgeistFlow(tracing=tracing == "true" if tracing in ("true", "false") else None)
    print(json.dumps(flow.kickoff()))


def run_with_trigger():
    """Standard AMP trigger entry point. The platform supplies a JSON payload."""
    from agentgeist.cli import utf8_console
    utf8_console()
    if len(sys.argv) != 2:
        raise ValueError("Provide exactly one JSON trigger payload")
    payload = json.loads(sys.argv[1])
    if not isinstance(payload, dict):
        raise ValueError("Trigger payload must be a JSON object")
    RunOptions.model_validate(payload)
    tracing = os.getenv("CREWAI_TRACING_ENABLED", "").lower()
    flow = AgentgeistFlow(tracing=tracing == "true" if tracing in ("true", "false") else None)
    print(json.dumps(flow.kickoff(inputs={"crewai_trigger_payload": payload})))


if __name__ == "__main__":
    kickoff()
