"""Email each tick's log to the operator through One (Gmail): a short explanation plus the full agent logs.

Set AGENTGEIST_REPORT_EMAIL to turn it on. One holds the Gmail grant, so this code never sees a Gmail
credential. Runs after submission and Discord, so it never delays a scored answer. Never retried: One's
send action is not idempotent, and a resend is a second email. Failures are recorded, never raised.
"""

import html
import json
import os
from pathlib import Path

import httpx

from .connections import describe_error

PASSTHROUGH = "https://api.withone.ai/v1/passthrough"
# One's "Send Email" action (POST /v1/gmail/send-email); override if One reissues it.
SEND_ACTION = "conn_mod_def::GGXAjWkZO8U::uMc1LQIHTTKzeMm3rLL5gQ"


def gmail_connection(client, secret):
    configured = os.getenv("ONE_GMAIL_CONNECTION_KEY", "").strip()
    if configured:
        return configured
    r = client.get("https://api.withone.ai/v1/vault/connections", headers={"x-one-secret": secret})
    r.raise_for_status()
    keys = [c["key"] for c in r.json().get("rows", []) if c.get("platform") == "gmail"]
    if len(keys) != 1:
        raise RuntimeError(f"Expected one Gmail connection in One, found {len(keys)}")
    return keys[0]


def explanation(run, outcome):
    """A few plain sentences: what the swarm saw, what it decided and why, and how the venue judged it."""
    if run.get("error"):
        return f"The swarm stopped on tick {run['tick_id']} before deciding. Error: {describe_line(run['error'])}."
    final, proposed = run["decision"], run["proposed"]
    text = (f"On tick {run['tick_id']} the swarm checked {len(run['sectors'])} sector(s) "
            f"({', '.join(run['sectors'])}) and decided to {final['action'].upper()}: {final['rationale'].rstrip('. ')}. ")
    text += ("The supervisor's safety check agreed with the commander. " if not run["supervisor_override"] else
             f"The commander proposed {proposed['action']}, but the supervisor's safety check overrode it. ")
    text += f"It took {run['swarm_seconds']} s and about {run['credits_reserved']} credits across {len(run['calls'])} model calls."
    for d in run.get("degraded", []):
        text += f" It continued without {d['service']} ({describe_line(d['error'])})."
    judgment = (outcome or {}).get("judgment")
    if judgment:
        text += (f" The venue judged it {judgment.get('verdict')} (score {judgment.get('score')}, "
                 f"false alarm: {judgment.get('false_alarm')}).")
    else:
        text += " It was not submitted to the venue (test or fixture run)."
    return text


def describe_line(error):
    if not isinstance(error, dict):
        return str(error)
    line = error.get("type", "error")
    if "status" in error:
        line += f" HTTP {error['status']}" + (f" from {error['service']}" if "service" in error else "")
    return line + (f": {error['reason']}" if "reason" in error else "")


def subject(run, outcome):
    action = run["decision"]["action"].upper() if run.get("decision") else "STOPPED"
    judgment = (outcome or {}).get("judgment") or {}
    verdict = f", venue: {judgment.get('verdict')} {judgment.get('score')}" if judgment else ""
    team = f" ({outcome['team']})" if outcome and outcome.get("team") else ""
    return f"Agentgeist{team} tick {run['tick_id']}: {action}{verdict}"  # no CR/LF: One refuses those in headers


def body(folder, run, outcome):
    e = html.escape  # agent answers and web lore are untrusted text: never let them become markup
    parts = [f"<h2>{e(subject(run, outcome))}</h2>", f"<p>{e(explanation(run, outcome))}</p>",
             "<h3>What each agent said</h3>",
             "<table border='1' cellpadding='4' cellspacing='0'><tr><th>Agent</th><th>Model call</th><th>Output</th></tr>"]
    parts += [f"<tr><td>{e(c['role'])}</td><td>{c['seconds']} s</td><td>{e(c['output']).replace(chr(10), '<br>')}</td></tr>"
              for c in run["calls"]]
    parts.append("</table>")
    discord = (outcome or {}).get("discord") or {}
    if discord.get("message_url"):
        parts.append(f"<p>Discord alert: <a href='{e(discord['message_url'])}'>{e(discord['message_url'])}</a></p>")
    parts.append("<h3>Full logs</h3>")
    for path in sorted(folder.glob("*.md")):
        if path.name != "summary.md":
            parts += [f"<h4>{e(path.name)}</h4>",
                      f"<pre style='white-space:pre-wrap;font-size:12px'>{e(path.read_text(encoding='utf-8'))}</pre>"]
    parts.append(f"<p style='color:#666'>Sent by the Agentgeist swarm through One (Gmail). Log folder: {e(str(folder))}</p>")
    return "\n".join(parts)


def email_report(folder):
    """Send the tick folder's report if AGENTGEIST_REPORT_EMAIL is set. Returns the result dict, or None."""
    to = os.getenv("AGENTGEIST_REPORT_EMAIL", "").strip()
    if not folder or not to:
        return None
    folder = Path(folder)
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    outcome_path = folder / "5_outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8")) if outcome_path.exists() else None
    try:
        secret = os.environ["ONE_API_KEY"]
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            connection = gmail_connection(client, secret)
            r = client.post(PASSTHROUGH + "/v1/gmail/send-email",
                            headers={"x-one-secret": secret, "x-one-connection-key": connection,
                                     "x-one-action-id": os.getenv("ONE_GMAIL_SEND_ACTION_ID", SEND_ACTION)},
                            json={"connectionKey": connection, "to": to, "subject": subject(run, outcome),
                                  "body": body(folder, run, outcome), "isHtml": True})
            r.raise_for_status()
            sent = r.json().get("email", {})
        result = {"status": "sent" if sent.get("sent") else "unconfirmed", "to": to, "via": "One (Gmail)",
                  "message_id": sent.get("messageId")}
    except Exception as exc:
        result = {"status": "failed", "to": to, "via": "One (Gmail)", "error": describe_error(exc)}
    (folder / "6_email.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    line = (f"- Email report: {result['status']} to {to} via One (Gmail)"
            + (f": {describe_line(result['error'])}" if "error" in result else ""))
    with open(folder / "summary.md", "a", encoding="utf-8") as f:
        f.write("\n" + line + "\n")
    return result
