"""Show what each agent saw and said, from the AGENTGEIST_TRACE file.

    uv run python scripts/show_trace.py              # last tick in the trace
    uv run python scripts/show_trace.py --tick 142   # a specific tick
    uv run python scripts/show_trace.py --all        # every tick
    uv run python scripts/show_trace.py --follow     # watch a run live (Ctrl+C to stop)
"""

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

from dotenv import load_dotenv

WIDTH = 100


def wrap(text, indent):
    lines = str(text).strip().splitlines() or [""]
    return "\n".join(textwrap.fill(line, WIDTH, initial_indent=indent, subsequent_indent=indent) for line in lines)


def render(e):
    head = f"[{e['at']} +{e['t']:5.2f}s] tick {e['tick_id']}"
    kind = e["kind"]
    if kind == "tick_start":
        return f"\n{'=' * WIDTH}\n{head}  START  sectors: {', '.join(e['sectors'])}  " \
               f"(budget {e['max_agents']} agents, {e['credits']} credits)"
    if kind == "tool":
        return f"{head}  {e['role']:<20} ran tool {e['tool']} ({e['seconds']}s)"
    if kind == "model_call":
        return (f"{head}  {e['role']:<20} model call {e['seconds']}s, {e['credits']} credits\n"
                f"{wrap('saw: ' + e['data_in'], '      ')}\n{wrap('said: ' + str(e['answer']), '      ')}")
    if kind == "decision":
        p, f = e["proposed"], e["final"]
        verdict = "SUPERVISOR OVERRODE the model" if e["supervisor_override"] else "supervisor agreed"
        return (f"{head}  DECISION {f['action'].upper()}  (model proposed {p['action']}; {verdict}; "
                f"{e['credits_reserved']} credits total)\n{wrap('why: ' + f['rationale'], '      ')}")
    if kind == "error":
        return f"{head}  ERROR {e['type']}"
    return f"{head}  {kind} {json.dumps(e)}"


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Readable view of the agent trace")
    parser.add_argument("path", nargs="?", default=os.getenv("AGENTGEIST_TRACE") or "state/agent-trace.jsonl")
    parser.add_argument("--tick", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    path = Path(args.path)
    if args.follow:
        print(f"Watching {path} for agent activity. Start a run in another terminal. Ctrl+C to stop.")
        offset = path.stat().st_size if path.exists() else 0
        try:
            while True:
                if path.exists() and path.stat().st_size > offset:
                    with open(path, encoding="utf-8") as f:
                        f.seek(offset)
                        for line in f:
                            if line.strip():
                                print(render(json.loads(line)), flush=True)
                        offset = f.tell()
                time.sleep(0.3)
        except KeyboardInterrupt:
            return

    if not path.exists():
        sys.exit(f"No trace at {path}. Set AGENTGEIST_TRACE (e.g. state/agent-trace.jsonl) and run a tick.")
    events = read(path)
    if not events:
        sys.exit("The trace is empty.")
    if not args.all:
        # The last run of the chosen tick: everything from its final tick_start onwards.
        tick = args.tick if args.tick is not None else events[-1]["tick_id"]
        starts = [i for i, e in enumerate(events) if e["kind"] == "tick_start" and e["tick_id"] == tick]
        if not starts:
            sys.exit(f"Tick {tick} is not in the trace.")
        events = [e for e in events[starts[-1]:] if e["tick_id"] == tick]
    for e in events:
        print(render(e))


if __name__ == "__main__":
    main()
