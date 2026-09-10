"""The store: one SQLite file every stage writes to and the GUI reads.

Stage 0 defines the `call` table, the log of every model call with its tokens
and cost. Later stages add their tables through `Store.migrate` with the same
pattern: `CREATE TABLE IF NOT EXISTS`, never a destructive change.

    from fx.store import Store
    s = Store("runs/demo/store.db")
    s.spent()                 # dollars so far, all models
    s.spent(model="gpt-5.6")  # one model
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_PATH = os.environ.get("FX_STORE", "runs/store.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS call (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,
    stage TEXT,
    note TEXT,
    model TEXT NOT NULL,
    base_url TEXT,
    prompt_sha TEXT NOT NULL,
    prompt_chars INTEGER,
    reply_chars INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost REAL NOT NULL DEFAULT 0,
    latency REAL,
    finish_reason TEXT,
    cached INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    reply TEXT
);
CREATE INDEX IF NOT EXISTS call_sha ON call(model, prompt_sha);
CREATE INDEX IF NOT EXISTS call_stage ON call(stage);
"""


class Store:
    def __init__(self, path: "str | Path" = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path), timeout=60, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.lock = threading.Lock()
        self.migrate(SCHEMA)

    def migrate(self, schema: str) -> None:
        with self.lock:
            self.con.executescript(schema)

    def insert(self, table: str, row: dict[str, Any]) -> int:
        keys = list(row)
        sql = f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})"
        with self.lock:
            cur = self.con.execute(sql, [_plain(row[k]) for k in keys])
            self.con.commit()
            return int(cur.lastrowid)

    def rows(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self.lock:
            return list(self.con.execute(sql, tuple(params)))

    def one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.rows(sql + " LIMIT 1", params)
        return rows[0] if rows else None

    # ---- calls
    def spent(self, model: Optional[str] = None, stage: Optional[str] = None) -> float:
        sql, params = "SELECT COALESCE(SUM(cost), 0) FROM call WHERE cached=0", []
        if model:
            sql += " AND model=?"; params.append(model)
        if stage:
            sql += " AND stage=?"; params.append(stage)
        return float(self.rows(sql, params)[0][0])

    def cached_reply(self, model: str, prompt_sha: str) -> Optional[str]:
        r = self.one("SELECT reply FROM call WHERE model=? AND prompt_sha=? AND error IS NULL AND reply IS NOT NULL AND reply != '' ORDER BY id DESC", (model, prompt_sha))
        return r["reply"] if r else None

    def spend_by_model(self) -> list[dict]:
        return [dict(r) for r in self.rows("SELECT model, COUNT(*) n, SUM(cached) cached, SUM(prompt_tokens) prompt_tokens, SUM(completion_tokens) completion_tokens, SUM(cost) cost FROM call GROUP BY model ORDER BY cost DESC")]


def _plain(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
