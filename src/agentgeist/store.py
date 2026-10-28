import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def process_lock(path):
    """One process per state database. Raises OSError at once if another process holds it."""
    with open(path, "a+") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield


class Store:
    """Durable tick inbox and submission outbox. Never advances the feed on retry."""

    def __init__(self, path="state/agentgeist.sqlite3"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS incidents (
            team TEXT, tick_id INTEGER, tick TEXT NOT NULL, decision TEXT,
            judgment TEXT, escalation TEXT, spent INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active', PRIMARY KEY(team,tick_id))""")
        self.db.execute("CREATE TABLE IF NOT EXISTS budgets(team TEXT PRIMARY KEY, ceiling INTEGER, spent INTEGER NOT NULL DEFAULT 0)")
        self.db.execute("CREATE TABLE IF NOT EXISTS ingress(id INTEGER PRIMARY KEY, team TEXT, payload TEXT NOT NULL)")
        self.db.commit()

    def save_tick(self, team, tick):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO incidents(team,tick_id,tick) VALUES(?,?,?)",
                            (team, tick.tick_id, tick.model_dump_json()))
            self.db.execute("INSERT OR IGNORE INTO budgets(team,ceiling) VALUES(?,?)",
                            (team, tick.swarm_budget.api_credits_remaining))

    def save_raw(self, team, payload):
        with self.db:
            cursor = self.db.execute("INSERT INTO ingress(team,payload) VALUES(?,?)", (team, json.dumps(payload)))
        return cursor.lastrowid

    def get(self, team, tick_id):
        self.db.row_factory = sqlite3.Row
        row = self.db.execute("SELECT * FROM incidents WHERE team=? AND tick_id=?", (team, tick_id)).fetchone()
        return dict(row) if row else None

    def status(self, team, tick_id, status):
        with self.db:
            self.db.execute("UPDATE incidents SET status=? WHERE team=? AND tick_id=?", (status, team, tick_id))

    def remaining(self, team):
        row = self.db.execute("SELECT ceiling-spent FROM budgets WHERE team=?", (team,)).fetchone()
        return row[0]

    def update(self, team, tick_id, column, value):
        if column not in {"decision", "judgment", "escalation"}:
            raise ValueError("Invalid outbox column")
        with self.db:
            self.db.execute(f"UPDATE incidents SET {column}=? WHERE team=? AND tick_id=?",
                            (json.dumps(value), team, tick_id))

    def pending(self, team):
        self.db.row_factory = sqlite3.Row
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM incidents WHERE team=? AND status='active' AND (judgment IS NULL OR escalation IS NULL "
            "OR json_extract(escalation,'$.status') IN ('pending','unknown')) ORDER BY tick_id", (team,))]

    def spent(self, team, tick_id):
        return self.db.execute("SELECT spent FROM incidents WHERE team=? AND tick_id=?", (team, tick_id)).fetchone()[0]

    def reserve(self, team, tick_id, units):
        from .budget import BudgetExceeded
        with self.lock, self.db:
            cursor = self.db.execute("UPDATE budgets SET spent=spent+? WHERE team=? AND ceiling-spent>=?",
                                     (units, team, units))
            if cursor.rowcount != 1:
                raise BudgetExceeded("Team-wide credit limit would be exceeded")
            self.db.execute("UPDATE incidents SET spent=spent+? WHERE team=? AND tick_id=?", (units, team, tick_id))
