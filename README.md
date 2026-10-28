# Agentgeist — The Exorcism Protocol

A supervised CrewAI hierarchy for Track 1. The team display name is **Agentgeist**.
The scenario API requires the lowercase handle `agentgeist`.

For CrewAI website runs and tracing, see [CrewAI setup](docs/CREWAI.md).
The standard `crewai run` command runs real agents on a fixture without venue submissions or Discord posts.

## Run

Use Python 3.10–3.13 and Node.js with `npx`.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
cp .env.example .env
# Fill the credentials in .env. Never commit .env.
.venv/bin/agentgeist demo
.venv/bin/agentgeist demo --fixture fixtures/control-tick.json
.venv/bin/agentgeist scoreboard
# Real CrewAI and provider calls; pulls one real tick but does not submit it:
.venv/bin/agentgeist run
# Explicit scored submissions and public Discord incident evidence:
.venv/bin/agentgeist run --ticks 2 --submit --discord
# Direct app-owned webhook delivery, including saved judgments without new ticks:
.venv/bin/agentgeist notify --tick-id 142 --discord-transport webhook
```

`demo` is an **offline policy demonstration**, not a simulated successful partner integration.
The fixtures come from the event's published fixture files. Demo mode never submits a decision.
`run` uses real providers. It stops if required specialist evidence is missing.

## Partial knowledge and supervision

| Role | Private input | Assigned tool |
|---|---|---|
| Sensor analyst | Sensors from every sector | Parasail embeddings and signature comparison |
| Sighting researcher | Sightings from every sector | You.com Search and Contents |
| Civilian marshal | Civilian status and containment state | Local civilian triage |
| Incident commander | Specialist reports | CrewAI hierarchical synthesis |
| Escalation officer | Approved decision and judgment only | One MCP or an operator-configured Discord webhook |

The three specialists do not receive the full tick. A mandatory sensing phase executes each specialist's tool before its model analysis.
The commander then coordinates their reports through a CrewAI hierarchical crew.
The manager has no delegation tools during synthesis. It cannot send another specialist's private data back to a worker.
The supervisor counts all four CrewAI agents before work starts. The escalation officer runs after the crew finishes.
Each specialist must actually use its assigned tool. The final decision includes tool evidence and any supervisor override.

The supervisor checks every sector. It prioritizes civilians at risk in a breached or corroborated dangerous sector.
It requires multiple physical indicators plus a credible sighting or breached containment before containment without evacuation.
It treats crowd motion and folklore alone as insufficient evidence. These transparent simulation thresholds are not real emergency guidance.

## Budget and failure behavior

- The active-agent gate rejects a tick with fewer than four available slots before model or partner calls.
- The shared ledger reserves credits before each model, embedding, research, and One escalation call.
- Direct webhook delivery reserves no incident credits, including failed setup checks and retries. The active-agent limit still applies.
- A durable team-wide ceiling also limits total reservations across ticks. New ticks cannot reset spent credits.
- The credit unit is a conservative local token-equivalent estimate, **not verified venue billing**. The feed does not document a monetary conversion.
- Model reservations use twice the cl100k input token count plus an output allowance.
- Model, embeddings and You.com calls retry only rate limits (429) and transient upstream errors (5xx, refused connections): at most twice, after 0.5 s and 1.5 s or a `Retry-After` of up to 4 s. Refused requests are not billed, so credits are reserved once. Timeouts are not retried. Retries appear in the agent trace and the tick log.
- Each specialist and the commander gets one model iteration. Invalid commander JSON triggers an explicit, recorded safety-policy fallback.
- HTTP requests have bounded timeouts. A failed tick remains in the durable inbox.
- SQLite saves each raw feed response before validation, each valid tick before inference, and each decision before submission.
- A restart processes pending ticks before it advances the team cursor.
- Submission replay uses the original tick and action. The feed documents one accepted answer per team per tick.
- Discord delivery requires a returned message identifier. An ambiguous delivery stays pending for manual inspection.
- Pre-send Discord failures remain retryable. Only the actual send marks delivery as unknown. Neither state stops scored processing.
- Model failures and rejected submissions enter terminal states. An operator can retry or abandon them with `resolve`.
- An unrecognized submission receipt enters `receipt_unverified`. The raw receipt remains available for inspection.
- A resource-limit failure uses a clearly labeled, zero-model `budget_fallback` safety decision. It never starts excess agents.
- `observe` does not post an incident alert. This reduces venue noise and false alarms.

The default database is `state/agentgeist.sqlite3`. Keep this database between runs.
Do not run multiple CLI processes for the same team at once.
The CLI enforces one local process per database with a file lock.
Use the same database for the same team. A new database starts a new local estimate and does not preserve prior spend.

### Resolve a saved incident

```bash
# Retry a terminal model failure without resetting its spent credits:
.venv/bin/agentgeist resolve --tick-id 142 --resolution retry
# After checking Discord and confirming that the attempted message does not exist:
.venv/bin/agentgeist resolve --tick-id 142 --resolution retry --confirm-no-discord-message
# Record a message identifier you verified manually. This is user-confirmed, not tool-verified:
.venv/bin/agentgeist resolve --tick-id 142 --resolution delivered --message-id 123456789012345678
# Stop retrying a rejected or otherwise unusable tick:
.venv/bin/agentgeist resolve --tick-id 142 --resolution abandon
```

Raw responses remain in the SQLite `ingress` table when validation fails. Inspect and repair contract drift before another pull.
Plain `run` advances the live cursor even without `--submit`. Use `demo` for offline checks.
Public messages use only the supervisor's trusted rationale. They include a simulation label and disable all mentions.

### Tick logs and email reports

Every tick writes `logs/<time>_tick-<id>/`: the raw tick, one file per agent (instructions, data given, thinking,
output, exact messages, timing, credits), `summary.md` and `run.json`. Live submissions add the venue verdict and Discord link.
`uv run python scripts/show_trace.py --follow` watches agents live when `AGENTGEIST_TRACE` is set.

Set `AGENTGEIST_REPORT_EMAIL` to email each tick's report: a short explanation, a table of what each agent said and the
full agent logs. **One** sends it through the operator's connected Gmail (`one add gmail`), so the swarm never holds a
Gmail credential. It runs after submission and Discord, is never retried (a resend is a duplicate email), and records
`6_email.json` plus a line in `summary.md`. A failure never stops the tick.

### Discord webhook delivery

Set `DISCORD_WEBHOOK_URL` in the environment or the ignored `.env` file. Never commit the URL or pass it as a command argument.
Set `DISCORD_TRANSPORT=webhook` to post directly when One is unavailable.
The app sends the notification from its normal scored-tick flow; no separate posting script is needed.

```bash
# Uses the webhook for each containment or evacuation judgment:
.venv/bin/agentgeist run --submit --discord --discord-transport webhook
# Drain saved accepted judgments only. This command never contacts the scenario feed:
.venv/bin/agentgeist notify --discord-transport webhook
```

The default `auto` transport prefers One. It uses the configured webhook only if One fails **before** a send attempt.
The `one` transport never uses the webhook. The `webhook` transport does not require One credentials.
One attempts retain the 1500-credit reservation. An automatic webhook fallback adds no further reservation.
If `auto` cannot reserve enough credits for One, it can still use a configured webhook without calling One.
Existing recorded reservations are not refunded or rewritten by this change.
Before a webhook post, the app verifies that the webhook targets `1564842098740695100`.
It requests the message receipt and records the returned message identifier and link.
The app skips already delivered judgments. An uncertain write stays `unknown`; neither automatic retries nor cross-transport fallback can duplicate that write.
Discord webhooks do not provide the same nonce-based duplicate protection as bot messages. Keep the same state database and resolve uncertain deliveries manually.
The `notify` command exits with an error when delivery remains pending or unknown.
These commands are bounded runs, not an installed background service.

## Connections

The scenario connection uses `MCPServerHTTP(url=..., streamable=True, headers={})`, not URL shorthand.
The scenario endpoint normally needs no key. Set `SCENARIO_TOKEN` only if organizers enable team tokens.

One's hosted endpoint uses OAuth, not an API key as a bearer token.
This project uses the official `@withone/mcp@2.3.2` stdio server with `ONE_SECRET=ONE_API_KEY`.
It discovers the Discord action and its request contract before it sends anything.
No Discord bot token is required. An optional venue webhook supports direct delivery with the operator's permission.

The configured `ONE_MCP_URL` remains available for an OAuth-capable client, but this CLI does not use it with the API key.
The default models were listed by the live Parasail catalog during setup. Override the model names in `.env` if needed.
Search and Contents use `X-API-Key`. CrewAI tracing is disabled by default.

## Verify

```bash
.venv/bin/python -m pytest -q
```

Source guide: https://youdotcom-oss.github.io/halloween-hack-2026/
