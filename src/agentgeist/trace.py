"""Opt-in record of what each agent saw and said. Set AGENTGEIST_TRACE to a .jsonl path.

One JSON object per line: tool runs, every model call (data in, answer out, seconds, credits) and the
final decision. Model prompts and answers only: no keys, URLs with credentials or MCP traffic.
View with `python scripts/show_trace.py` (add --follow to watch a run live).
"""

import json
import os
import threading
import time
from pathlib import Path


class Trace:
    def __init__(self, tick_id):
        self.path = os.getenv("AGENTGEIST_TRACE", "").strip()
        self.tick_id = tick_id
        self.start = time.monotonic()
        self.lock = threading.Lock()  # specialists write from parallel threads
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind, **data):
        if not self.path:
            return
        line = json.dumps({"tick_id": self.tick_id, "at": time.strftime("%H:%M:%S"),
                           "t": round(time.monotonic() - self.start, 2), "kind": kind, **data},
                          ensure_ascii=False, default=str)
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
