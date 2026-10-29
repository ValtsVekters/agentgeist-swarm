"""Local dashboard: run test or live ticks, watch the agents live, and read every tick's logs.

    uv run agentgeist-ui            # opens http://127.0.0.1:8765
    uv run agentgeist-ui --no-browser --port 8765

Serves only on 127.0.0.1. Test ticks run in this process (warmed up once, so they show the real decision time).
Live ticks run the normal `agentgeist run` CLI as a child process, with its database, lock and safety checks.
The page never receives keys or the webhook URL: it sees tick data, agent prompts and answers, and results.
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from .contracts import team_handle

SCENARIOS = ("outbreak-level-5", "control-tick")
LIVE_EVENTS = ("decision", "judgment", "discord_receipt", "escalation_pending", "escalation_skipped", "tick_failed",
               "email_report", "submission_rejected", "receipt_unverified", "error", "scoreboard")


class Runner:
    """One run at a time; the page polls status, live agent events and live CLI output."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = None  # {"kind", "label", "started"}
        self.last = None     # last finished run summary
        self.warm = "warming up"
        self.output = collections.deque(maxlen=300)

    def busy(self):
        return self.current is not None

    def start(self, kind, label, target):
        with self.lock:
            if self.current:
                raise RuntimeError("A run is already in progress")
            self.current = {"kind": kind, "label": label, "started": time.time()}
            self.output.clear()
        threading.Thread(target=self._wrap, args=(target,), daemon=True).start()

    def _wrap(self, target):
        try:
            self.last = {**self.current, **target(), "finished": time.time()}
        except Exception as exc:
            from .connections import describe_error
            self.last = {**self.current, "ok": False, "error": describe_error(exc), "finished": time.time()}
        finally:
            with self.lock:
                self.current = None


RUNNER = Runner()


def without_email(enabled):
    """Child environment and in-process toggle for the email report (one run at a time, so no race)."""
    env = dict(os.environ)
    if not enabled:
        env.pop("AGENTGEIST_REPORT_EMAIL", None)
    return env


def run_fixture(scenario, email):
    from .main import RunOptions, execute
    saved = os.environ.get("AGENTGEIST_REPORT_EMAIL")
    if not email:
        os.environ.pop("AGENTGEIST_REPORT_EMAIL", None)
    try:
        result = execute(RunOptions(mode="fixture", scenario=scenario))
    finally:
        if saved is not None:
            os.environ["AGENTGEIST_REPORT_EMAIL"] = saved
    run = json.loads((Path(result["log_dir"]) / "run.json").read_text(encoding="utf-8")) if result.get("log_dir") else {}
    return {"ok": True, "action": result["decision"]["action"], "seconds": run.get("swarm_seconds"),
            "log": Path(result["log_dir"]).name if result.get("log_dir") else None, "email": result.get("email_report")}


def run_live(team, scenario, submit, discord, email):
    cmd = [sys.executable, "-m", "agentgeist.cli", "--team", team, "--db", f"state/{team}.sqlite3",
           "run", "--scenario", scenario, "--ticks", "1"]
    if submit:
        cmd.append("--submit")
    if discord:
        cmd += ["--discord", "--discord-transport", os.getenv("DISCORD_TRANSPORT", "webhook")]
    env = {**without_email(email), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=env)
    summary = {"ok": False}
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue  # CrewAI console noise; the CLI's JSON events carry the results
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") in LIVE_EVENTS:
            RUNNER.output.append(event)
        if event.get("event") == "decision":
            summary.update(ok=True, action=event["decision"]["action"], seconds=event.get("decision_seconds"),
                           log=Path(event["log_dir"]).name if event.get("log_dir") else None)
        elif event.get("event") == "judgment":
            summary["judgment"] = event["judgment"]
        elif event.get("event") == "tick_failed":
            summary.update(ok=False, error={k: v for k, v in event.items() if k not in ("event", "tick_id")})
    summary["exit_code"] = proc.wait()
    return summary


def trace_path():
    return Path(os.getenv("AGENTGEIST_TRACE") or "state/agent-trace.jsonl")


def log_root():
    return Path(os.getenv("AGENTGEIST_LOG_DIR", "logs") or "logs").resolve()


def list_runs(limit=60):
    runs = []
    for run_json in sorted(log_root().rglob("run.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        folder = run_json.parent
        try:
            run = json.loads(run_json.read_text(encoding="utf-8"))
            outcome = folder / "5_outcome.json"
            outcome = json.loads(outcome.read_text(encoding="utf-8")) if outcome.exists() else {}
        except ValueError:
            continue
        judgment = outcome.get("judgment") or {}
        runs.append({"id": str(folder.relative_to(log_root())).replace("\\", "/"), "tick": run.get("tick_id"),
                     "started": run.get("started"), "action": (run.get("decision") or {}).get("action"),
                     "seconds": run.get("swarm_seconds"), "error": run.get("error"), "team": outcome.get("team"),
                     "verdict": judgment.get("verdict"), "score": judgment.get("score"),
                     "override": run.get("supervisor_override")})
    return runs


def read_run(run_id):
    root = log_root()
    folder = (root / run_id).resolve()
    if root not in folder.parents or not folder.is_dir():  # stay inside logs/: no path traversal
        raise FileNotFoundError(run_id)
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix in (".md", ".json")}


def read_events(offset):
    path = trace_path()
    if not path.exists():
        return {"offset": 0, "events": []}
    size = path.stat().st_size
    if offset < 0:
        return {"offset": size, "events": []}  # the page starts at the end: only new activity
    if offset > size:
        offset = 0  # the trace file was replaced
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    complete = chunk.rfind(b"\n") + 1  # never hand back a half-written line
    lines = chunk[:complete].decode("utf-8", errors="replace").splitlines()
    return {"offset": offset + complete, "events": [json.loads(l) for l in lines if l.strip()]}


def scoreboard():
    from .connections import MCPConnection
    connection = MCPConnection(os.environ["SCENARIO_MCP"])
    try:
        return connection.call("hack26-get-scoreboard")
    finally:
        connection.close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the terminal quiet; requests carry no secrets but are noisy

    def send(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/":
                self.send(200, files("agentgeist").joinpath("dashboard.html").read_bytes(), "text/html")
            elif url.path == "/api/status":
                self.send(200, {"running": RUNNER.current, "last": RUNNER.last, "warm": RUNNER.warm,
                                "output": list(RUNNER.output), "model": os.getenv("PARASAIL_CHAT_MODEL"),
                                "email": bool(os.getenv("AGENTGEIST_REPORT_EMAIL")), "team": os.getenv("TEAM_HANDLE"),
                                "discord_transport": os.getenv("DISCORD_TRANSPORT", "auto")})
            elif url.path == "/api/events":
                self.send(200, read_events(int(query.get("offset", ["0"])[0])))
            elif url.path == "/api/runs":
                self.send(200, list_runs())
            elif url.path == "/api/run":
                self.send(200, read_run(query["id"][0]))
            elif url.path == "/api/scoreboard":
                self.send(200, scoreboard())
            else:
                self.send(404, {"error": "not found"})
        except FileNotFoundError:
            self.send(404, {"error": "not found"})
        except Exception as exc:
            from .connections import describe_error
            self.send(500, {"error": describe_error(exc)})

    def do_POST(self):
        if urlparse(self.path).path != "/api/start":
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            scenario = body.get("scenario")
            if scenario not in SCENARIOS:
                raise ValueError("Unknown scenario")
            email = bool(body.get("email"))
            if body.get("kind") == "fixture":
                RUNNER.start("fixture", f"Test tick: {scenario}", lambda: run_fixture(scenario, email))
            elif body.get("kind") == "live":
                if body.get("confirm") != "LIVE":
                    raise ValueError("Live runs need explicit confirmation")
                team = team_handle(body.get("team", ""))
                submit, discord = bool(body.get("submit")), bool(body.get("discord"))
                RUNNER.start("live", f"Live tick: {team} on {scenario}",
                             lambda: run_live(team, scenario, submit, discord, email))
            else:
                raise ValueError("Unknown run kind")
            self.send(200, {"started": True})
        except (ValueError, RuntimeError) as exc:
            self.send(400, {"error": str(exc)})


def main():
    load_dotenv()
    os.environ.setdefault("AGENTGEIST_TRACE", "state/agent-trace.jsonl")  # the live view reads this stream
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    parser = argparse.ArgumentParser(description="Agentgeist dashboard")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def warm():
        from .swarm import warm_up
        try:
            warm_up()
            RUNNER.warm = "ready"
        except Exception as exc:
            RUNNER.warm = f"warm-up failed: {type(exc).__name__}"
    threading.Thread(target=warm, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Agentgeist dashboard on {url} (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
