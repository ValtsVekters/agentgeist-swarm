"""Time where a tick's decision_seconds goes. Fixture only: no venue feed, no submission, no Discord.

    uv run python scripts/profile_tick.py [outbreak-level-5|control-tick]

Wraps the swarm's phases and its network calls with timers; app code is not modified.
Spends real Parasail and You.com credits, like `uv run kickoff`.
"""

import json
import re
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from importlib.resources import files

from dotenv import load_dotenv

load_dotenv()

import_start = time.perf_counter()
import httpx  # noqa: E402
from crewai import Agent  # noqa: E402

from agentgeist import swarm  # noqa: E402
from agentgeist.budget import Ledger  # noqa: E402
from agentgeist.cli import utf8_console  # noqa: E402
from agentgeist.contracts import Tick  # noqa: E402
import_seconds = time.perf_counter() - import_start

phases = []                    # (label, start, end), outermost per thread only
calls = defaultdict(list)      # leaf label -> [seconds, ...]
local = threading.local()      # specialists run in parallel threads
lock = threading.Lock()


@contextmanager
def timed(bucket, label):
    start = time.perf_counter()
    if bucket is phases:
        local.depth = getattr(local, "depth", 0) + 1
    try:
        yield
    finally:
        end = time.perf_counter()
        with lock:
            if bucket is phases:
                local.depth -= 1
                if local.depth == 0:  # a phase nested in another (cached tool re-read) must not double count
                    phases.append((label, start, end))
            else:
                calls[label].append(end - start)


def wrap(owner, name, label_for, bucket):
    original = getattr(owner, name)

    def timed_call(*args, **kwargs):
        with timed(bucket, label_for(*args, **kwargs)):
            return original(*args, **kwargs)

    setattr(owner, name, timed_call)


def http_label(client, url, *a, **k):
    url = str(url)
    if "parasail" in url:
        return "HTTP Parasail " + url.rsplit("/", 1)[-1]
    if "ydc-index" in url:
        return "HTTP You.com " + url.rsplit("/", 1)[-1]
    return "HTTP other"


prompt_chars = defaultdict(int)  # LLM label -> characters sent across its calls


def llm_label(llm, messages, *a, **k):
    text = json.dumps(messages, default=str)
    role = re.search(r"You are ([A-Za-z ]+?)[.\\]", text)
    label = f"LLM Parasail ({role.group(1) if role else 'unknown role'})"
    prompt_chars[label] += len(text)
    return label


def main():
    utf8_console()
    scenario = sys.argv[1] if len(sys.argv) > 1 else "outbreak-level-5"
    data = json.loads(files("agentgeist").joinpath("fixtures", scenario + ".json").read_text(encoding="utf-8"))
    tick = Tick.model_validate(data["ticks"][0])

    wrap(httpx.Client, "post", http_label, calls)
    wrap(swarm.BudgetLLM, "call", llm_label, calls)
    wrap(swarm.ScopedTool, "_run", lambda tool, *a, **k: f"tool {tool.name}", phases)
    wrap(Agent, "execute_task", lambda agent, *a, **k: f"specialist {agent.role}", phases)
    wrap(swarm.ScopedCrew, "kickoff", lambda *a, **k: "commander crew (hierarchical)", phases)

    warm_start = time.perf_counter()
    swarm.warm_up()  # as the CLI does before pulling ticks
    print(f"warm-up (before the tick, not counted): {time.perf_counter() - warm_start:.2f}s")
    ledger = Ledger(tick.swarm_budget.max_active_agents, tick.swarm_budget.api_credits_remaining)
    start = time.perf_counter()
    result = swarm.run_swarm(tick, ledger)
    total = time.perf_counter() - start

    print(f"\nscenario {scenario}, tick {tick.tick_id}: action={result['decision']['action']}, "
          f"model proposed={result['proposed']['action']}, supervisor override={result['supervisor_override']}, "
          f"credits={result['credits_reserved']}")
    print(f"commander rationale: {result['proposed']['rationale']}")
    print(f"imports (crewai etc., before the clock starts): {import_seconds:5.1f}s\n")
    print(f"{'PHASE (by start time; overlap = parallel)':45} {'start':>6} {'end':>6} {'seconds':>8}")
    covered, reach = 0.0, start
    for label, s, e in sorted(phases, key=lambda p: p[1]):
        print(f"{label:45} {s - start:6.2f} {e - start:6.2f} {e - s:8.2f}")
        covered += max(e - max(s, reach), 0)
        reach = max(reach, e)
    print(f"{'outside any phase (setup, parsing, supervisor)':45} {'':13} {max(total - covered, 0):8.2f}")
    print(f"{'TOTAL run_swarm (wall clock)':45} {'':13} {total:8.2f}\n")

    print(f"{'NETWORK / MODEL CALL':45} {'count':>5} {'seconds':>8} {'share':>6} {'prompt chars':>13}")
    for label, times in sorted(calls.items(), key=lambda kv: -sum(kv[1])):
        chars = f"{prompt_chars[label]:13,}" if label in prompt_chars else ""
        print(f"{label:45} {len(times):5} {sum(times):8.2f} {sum(times) / total:6.0%} {chars}")


if __name__ == "__main__":
    main()
