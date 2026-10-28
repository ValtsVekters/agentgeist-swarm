"""Records what each agent was given, thought and answered.

Two outputs, both prompts and answers only (no keys, credential-bearing URLs or MCP traffic):
- AGENTGEIST_TRACE=<path.jsonl>: a running event stream; `python scripts/show_trace.py --follow` watches it.
- AGENTGEIST_LOG_DIR (default "logs", empty to disable): after each tick, one folder per tick with a
  file per agent (system prompt, data given, thinking, output) and a summary.
"""

import json
import os
import re
import threading
import time
from pathlib import Path

# File order in a tick folder; the commander synthesizes last.
AGENT_ORDER = ["Sensor analyst", "Sighting researcher", "Civilian marshal", "Incident commander"]


def split_answer(raw):
    """CrewAI's text protocol may put reasoning before 'Final Answer:'. Return (thinking, output)."""
    text = str(raw or "").strip()
    if "Final Answer:" in text:
        thinking, output = text.split("Final Answer:", 1)
        thinking = re.sub(r"^\s*Thought:\s*", "", thinking).strip()
        return thinking or None, output.strip()
    return None, text


def pretty_data(prompt):
    data = prompt.split("Data: ", 1)[-1].split("\n", 1)[0]
    try:
        return json.dumps(json.loads(data), indent=2, ensure_ascii=False), "json"
    except ValueError:
        return data, ""


class Trace:
    def __init__(self, tick_id):
        self.path = os.getenv("AGENTGEIST_TRACE", "").strip()
        self.log_root = os.getenv("AGENTGEIST_LOG_DIR", "logs").strip()
        self.tick_id = tick_id
        self.started = time.strftime("%Y-%m-%d %H:%M:%S")
        self.start = time.monotonic()
        self.calls = []
        self.retries = []
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

    def record_call(self, messages, answer, seconds, credits, model):
        system = messages[0]["content"] if messages else ""
        prompt = messages[-1]["content"] if messages else ""
        role = re.search(r"You are (.+?)\. ", system)
        role = role.group(1) if role else "unknown"
        with self.lock:
            self.calls.append({"role": role, "messages": messages, "answer": answer, "seconds": seconds,
                               "credits": credits, "model": model, "t": round(time.monotonic() - self.start, 2)})
        self.event("model_call", role=role, seconds=seconds, credits=credits,
                   data_in=prompt.split("Data: ", 1)[-1].split("\n", 1)[0], answer=answer)

    def record_retry(self, service, status, attempt, wait):
        retry = {"service": service, "status": status, "attempt": attempt, "wait": wait,
                 "t": round(time.monotonic() - self.start, 2)}
        with self.lock:
            self.retries.append(retry)
        self.event("retry", **{k: v for k, v in retry.items() if k != "t"})

    def save_run(self, tick, result=None, evidence=None, error=None):
        """Write logs/<time>_tick-<id>/ and return its path, or None when disabled."""
        if not self.log_root:
            return None
        folder = Path(self.log_root) / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_tick-{self.tick_id}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "0_tick.json").write_text(tick.model_dump_json(indent=2), encoding="utf-8")
        evidence = evidence or {}
        calls = sorted(self.calls, key=lambda c: (AGENT_ORDER.index(c["role"]) if c["role"] in AGENT_ORDER else 9, c["t"]))
        for call in calls:
            index = AGENT_ORDER.index(call["role"]) + 1 if call["role"] in AGENT_ORDER else 9
            name = f"{index}_{call['role'].lower().replace(' ', '_')}.md"
            (folder / name).write_text(self._agent_page(call, evidence.get(call["role"])), encoding="utf-8")
        (folder / "summary.md").write_text(self._summary(tick, result, calls, error), encoding="utf-8")
        return str(folder)

    def _agent_page(self, call, tool_evidence):
        system = call["messages"][0]["content"] if call["messages"] else ""
        prompt = call["messages"][-1]["content"] if call["messages"] else ""
        data, lang = pretty_data(prompt)
        thinking, output = split_answer(call["answer"])
        page = [f"# {call['role']} — tick {self.tick_id}", "",
                f"- Answered at +{call['t']}s after tick start, model call {call['seconds']}s",
                f"- Credits reserved: {call['credits']}", f"- Model: `{call['model']}`", "",
                "## Instructions (system prompt, the same every tick)", "", "```text", system.strip(), "```", "",
                "## Data given this tick", "", f"```{lang}", data, "```", ""]
        if tool_evidence is not None:
            page += ["## Raw tool evidence behind that data", "", "```json",
                     json.dumps(tool_evidence, indent=2, ensure_ascii=False, default=str), "```", ""]
        page += ["## Thinking", "", thinking or "_None: the model answered directly, with no separate reasoning._", "",
                 "## Output", "", "```text", output, "```", "",
                 "## Exact messages sent to the model", ""]
        for m in call["messages"]:
            page += [f"**{m['role']}**", "", "```text", str(m["content"]).strip(), "```", ""]
        return "\n".join(page)

    def _summary(self, tick, result, calls, error):
        lines = [f"# Tick {self.tick_id} — {self.started}", "",
                 f"Sectors: {', '.join(s.sector for s in tick.sector_reports)}", ""]
        if error:
            lines += [f"**Stopped with an error:** `{error}`", ""]
        if result:
            final, proposed = result["decision"], result["proposed"]
            lines += [f"## Decision: **{final['action'].upper()}**", "", f"> {final['rationale']}", "",
                      f"- Commander proposed: `{proposed['action']}`",
                      f"- Supervisor: {'**overrode** the commander' if result['supervisor_override'] else 'agreed'}",
                      f"- Model answer valid JSON: {result['model_decision_valid']}",
                      f"- Swarm time: {round(time.monotonic() - self.start, 2)}s, "
                      f"credits reserved: {result['credits_reserved']}, peak agents: {result['peak_active_agents']}", ""]
        lines += ["## What each agent said", "", "| Agent | At | Model call | Output |", "|---|---|---|---|"]
        for c in calls:
            output = split_answer(c["answer"])[1].replace("\n", " · ").replace("|", "/")
            lines.append(f"| {c['role']} | +{c['t']}s | {c['seconds']}s | {output} |")
        if self.retries:
            lines += ["", "## Retries (rate limits or temporary upstream errors)", "",
                      "| At | Service | Status | Retry | Waited |", "|---|---|---|---|---|"]
            lines += [f"| +{r['t']}s | {r['service']} | {r['status']} | {r['attempt']} | {r['wait']}s |"
                      for r in self.retries]
        return "\n".join(lines) + "\n"


def record_outcome(folder, *, team, judgment=None, escalation=None):
    """Append the venue's judgment and the Discord delivery to a tick folder after submission."""
    if not folder:
        return
    folder = Path(folder)
    (folder / "5_outcome.json").write_text(json.dumps({"team": team, "judgment": judgment, "discord": escalation},
                                                      indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["", "## Venue outcome", "", f"- Team: `{team}`"]
    if judgment:
        lines += [f"- Verdict: **{judgment.get('verdict')}**, score {judgment.get('score')}, "
                  f"false alarm: {judgment.get('false_alarm')} (expected `{judgment.get('expected')}`)",
                  f"- Venue rationale: {judgment.get('rationale')}"]
    if escalation:
        link = escalation.get("message_url")
        lines.append(f"- Discord: {escalation.get('status')}" + (f" — [message]({link})" if link else ""))
    with open(folder / "summary.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
