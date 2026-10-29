"""Local dashboard: run saved or random ticks, see each agent's answer and the decision, post it to Discord.

    uv run agentgeist-ui            # opens http://127.0.0.1:8765
    uv run agentgeist-ui --no-browser --port 8765

Serves only on 127.0.0.1. Ticks run in this warmed-up process and are never submitted to the venue. Contain and
evacuate decisions post to Discord #loft-incident as the team in TEAM_HANDLE, marked as practice (not submitted);
observe never posts. The page never receives keys or the webhook URL.
"""

import argparse
import hmac
import json
import os
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from .contracts import Tick, safety_floor, team_handle
from .scenarios import random_tick

SAVED = {"142": ("outbreak-level-5", 0), "143": ("outbreak-level-5", 1), "900": ("control-tick", 0)}
# Practice ticks are not submitted, so the Discord post says so instead of showing a venue score.
PRACTICE = {"verdict": "practice", "score": "practice (not submitted)", "false_alarm": "n/a"}


class Runner:
    """One run at a time; the page polls status and the live agent events."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = None  # {"label", "started"}
        self.last = None     # results of the last finished run
        self.warm = "warming up"

    def start(self, label, target):
        with self.lock:
            if self.current:
                raise RuntimeError("A run is already in progress")
            self.current = {"label": label, "started": time.time()}
        threading.Thread(target=self._wrap, args=(target,), daemon=True).start()

    def _wrap(self, target):
        try:
            self.last = target()
        except Exception as exc:
            from .connections import describe_error
            self.last = {"results": [], "error": describe_error(exc)}
        finally:
            with self.lock:
                self.current = None


RUNNER = Runner()


def saved_tick(tick_id):
    scenario, index = SAVED[tick_id]
    data = json.loads(files("agentgeist").joinpath("fixtures", scenario + ".json").read_text(encoding="utf-8"))
    tick = Tick.model_validate(data["ticks"][index])
    return tick, safety_floor(tick).action


def post_discord(team, tick, result, ledger):
    if result["decision"]["action"] == "observe":
        return {"status": "not posted (observe)"}
    from .connections import describe_error
    from .notifier import notify
    try:
        receipt = notify(team, tick, result, PRACTICE, ledger, transport=os.getenv("DISCORD_TRANSPORT", "webhook"))
        return {"status": "posted", "url": receipt.get("message_url")}
    except Exception as exc:
        return {"status": "failed", "error": describe_error(exc)}


def run_ticks(ticks, discord):
    """Run (tick, expected) items through the warmed swarm. Never submitted; optionally posted to Discord."""
    from .budget import Ledger
    from .connections import describe_error
    from .swarm import run_swarm, warm_up
    warm_up()
    team = team_handle(os.getenv("TEAM_HANDLE", "agentgeist"))
    results = []
    for tick, expected in ticks:
        row = {"tick": tick.tick_id, "expected": expected}
        try:
            ledger = Ledger(tick.swarm_budget.max_active_agents, tick.swarm_budget.api_credits_remaining)
            result = run_swarm(tick, ledger)
            run = json.loads((Path(result["log_dir"]) / "run.json").read_text(encoding="utf-8")) if result.get("log_dir") else {}
            result["decision_seconds"] = run.get("swarm_seconds")
            row.update(action=result["decision"]["action"], rationale=result["decision"]["rationale"],
                       matched=result["proposed"]["action"] == expected, log=result.get("log_dir"))
            if discord:
                row["discord"] = post_discord(team, tick, result, ledger)
        except Exception as exc:
            row.update(error=describe_error(exc), matched=False)
        results.append(row)
    return {"team": team, "results": results}


def trace_path():
    return Path(os.getenv("AGENTGEIST_TRACE") or "state/agent-trace.jsonl")


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


# Per-process secret, embedded in the served page and required to start a run: another site open in the
# browser can reach 127.0.0.1 but cannot read this page, so it cannot learn the token.
TOKEN = secrets.token_urlsafe(32)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the terminal quiet; requests carry no secrets but are noisy

    def origins(self):
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def trusted(self, write=False):
        """Refuse DNS rebinding (foreign Host) on every request, and cross-site requests on writes."""
        if self.headers.get("Host", "") not in self.origins():
            return False
        if not write:
            return True
        origin = self.headers.get("Origin")
        return (self.headers.get("Content-Type", "").startswith("application/json")  # forces a CORS preflight
                and (origin is None or (origin.startswith("http://") and urlparse(origin).netloc in self.origins()))
                and hmac.compare_digest(self.headers.get("X-Agentgeist-Token", ""), TOKEN))

    def send(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")  # no framing: nobody can trick a click
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.trusted():
            return self.send(403, {"error": "forbidden host"})
        url = urlparse(self.path)
        if url.path == "/":
            page = files("agentgeist").joinpath("dashboard.html").read_bytes()
            self.send(200, page.replace(b"__AGENTGEIST_TOKEN__", TOKEN.encode()), "text/html")
        elif url.path == "/api/status":
            self.send(200, {"running": RUNNER.current, "last": RUNNER.last, "warm": RUNNER.warm})
        elif url.path == "/api/events":
            self.send(200, read_events(int(parse_qs(url.query).get("offset", ["0"])[0])))
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        # Read the (small, capped) body before answering: replying first makes Windows abort the connection.
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > 65536:
            return self.send(413, {"error": "request too large"})
        raw = self.rfile.read(length)
        if not self.trusted(write=True):
            return self.send(403, {"error": "forbidden: cross-site or missing dashboard token"})
        if urlparse(self.path).path == "/api/email":
            # Emails the last run's tick reports through One (Gmail). Folders come from the server's own
            # results, never from the request, so the page cannot point this at other files.
            from .report import email_report
            if RUNNER.current or not RUNNER.last:
                return self.send(400, {"error": "Run a tick first"})
            sent = [email_report(r["log"]) for r in RUNNER.last["results"] if r.get("log")]
            ok = sum(1 for s in sent if s and s["status"] == "sent")
            return self.send(200, {"sent": ok, "total": len(sent), "configured": any(sent)})
        if urlparse(self.path).path != "/api/start":
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(raw or b"{}")
            discord = bool(body.get("discord"))
            if body.get("kind") == "saved":
                tick_id = str(body.get("tick"))
                if tick_id not in SAVED:
                    raise ValueError("Unknown saved tick")
                RUNNER.start(f"Tick {tick_id}", lambda: run_ticks([saved_tick(tick_id)], discord))
            elif body.get("kind") == "random":
                count = int(body.get("count", 1))
                if not 1 <= count <= 10:
                    raise ValueError("Run 1 to 10 random ticks at a time")
                seed = secrets.randbelow(10**6)
                ticks = [random_tick(seed + i) for i in range(count)]
                RUNNER.start(f"{count} random tick{'s' if count > 1 else ''}", lambda: run_ticks(ticks, discord))
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
