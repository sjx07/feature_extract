"""The store: one SQLite file every stage writes to and the GUI reads.

Every table of every stage is defined here, in dependency order, and created when
a Store opens: `CREATE TABLE IF NOT EXISTS`, never a destructive change. The stage
modules hold only the code that reads and writes them.

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
-- stage 0: every model call, priced, with the cache
CREATE TABLE IF NOT EXISTS call (
    id INTEGER PRIMARY KEY, at TEXT NOT NULL, stage TEXT, note TEXT, model TEXT NOT NULL, base_url TEXT,
    prompt_sha TEXT NOT NULL, prompt_chars INTEGER, reply_chars INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,
    cost REAL NOT NULL DEFAULT 0, latency REAL, finish_reason TEXT, cached INTEGER NOT NULL DEFAULT 0, error TEXT, reply TEXT,
    provider TEXT, billed REAL);
CREATE INDEX IF NOT EXISTS call_sha ON call(model, prompt_sha);
CREATE INDEX IF NOT EXISTS call_stage ON call(stage);

-- corpora and their prompts, as imported
CREATE TABLE IF NOT EXISTS corpus (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, source TEXT, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS prompt (
    id TEXT PRIMARY KEY, corpus INTEGER NOT NULL REFERENCES corpus(id), sha TEXT NOT NULL, text TEXT NOT NULL,
    domain TEXT, system TEXT, task TEXT, source_id TEXT, meta TEXT, at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS prompt_corpus ON prompt(corpus);
CREATE INDEX IF NOT EXISTS prompt_sha ON prompt(sha);

-- stage 1: decomposition. span holds every located node of a prompt's tree: sections, atoms, material,
-- unrefined leaves, and the gaps the model declined; note carries the material kind or the gap's outcome.
-- reading holds a leaf's facets: an atom's guidance, or for a material leaf the one facet saying what the prompt provides
-- (join span.kind to tell them apart). The unit later stages assign features to. decomp is the stage's record per prompt.
CREATE TABLE IF NOT EXISTS span (
    id INTEGER PRIMARY KEY, prompt TEXT NOT NULL REFERENCES prompt(id), path TEXT NOT NULL, lo INTEGER NOT NULL, hi INTEGER NOT NULL,
    kind TEXT NOT NULL, note TEXT, flags TEXT, start TEXT, end TEXT, depth INTEGER);
CREATE INDEX IF NOT EXISTS span_prompt ON span(prompt);
CREATE TABLE IF NOT EXISTS reading (
    id INTEGER PRIMARY KEY, prompt TEXT NOT NULL REFERENCES prompt(id), span INTEGER NOT NULL REFERENCES span(id),
    verb TEXT, object TEXT, qualifier TEXT, polarity TEXT, condition TEXT, domain_terms TEXT, declaration TEXT);
CREATE INDEX IF NOT EXISTS reading_prompt ON reading(prompt);
CREATE TABLE IF NOT EXISTS decomp (
    prompt TEXT PRIMARY KEY REFERENCES prompt(id), status TEXT NOT NULL, model TEXT, coverage REAL, material_share REAL,
    calls INTEGER, seconds REAL, reasks INTEGER, flags TEXT, failures TEXT, error TEXT, at TEXT NOT NULL);

-- stage 2: the feature library of a corpus, one per kind (guidance readings, material readings).
-- realization: one distinct declaration (polarity + normalised wording); reading.realization points at it, so
--   support is a join. Assignment works on realizations, never on single readings.
-- feature: the codebook as a tree in one table: level 'group', 'feature' and 'variant' rows, parent = the row above
--   (a variant is a feature narrowed by a constraint). The tree only grows: a node is never rewritten, and `round`
--   says when it was added. Realizations are assigned to features and variants; a group's support is the union.
-- vector: one embedding per realization, the retrieval that sorts the leftovers (see fx.library.cluster).
-- assignment: realization -> node (NULL = open) under one codebook, with the model's confidence; note says
--   'specific' (no other prompt says the same thing yet) or 'named' (placed by the naming call that made its node).
-- flag: what the read-only coherence judge reported: a misfit member, a split, or two siblings it could not tell
--   apart. A flag raised for the first time is actionable (reopen sends the member open); one raised again on the
--   same member and node after that is standing: the member stays and the flag is the report.
-- codebook: the version record: which model wrote it, from which round, and the anchor agreement measured on it.
CREATE TABLE IF NOT EXISTS realization (
    id INTEGER PRIMARY KEY, corpus INTEGER NOT NULL REFERENCES corpus(id), kind TEXT NOT NULL, key TEXT NOT NULL,
    polarity TEXT NOT NULL, declaration TEXT NOT NULL, n INTEGER NOT NULL, prompts INTEGER NOT NULL, conditions TEXT, head TEXT,
    sample TEXT, domain_terms TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS realization_key ON realization(corpus, kind, key);
CREATE TABLE IF NOT EXISTS codebook (
    id INTEGER PRIMARY KEY, corpus INTEGER NOT NULL REFERENCES corpus(id), kind TEXT NOT NULL, version INTEGER NOT NULL,
    model TEXT, round INTEGER NOT NULL, notes TEXT, anchor_agreement REAL, at TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS codebook_version ON codebook(corpus, kind, version);
CREATE TABLE IF NOT EXISTS feature (
    id INTEGER PRIMARY KEY, codebook INTEGER NOT NULL REFERENCES codebook(id), level TEXT NOT NULL, parent INTEGER REFERENCES feature(id),
    prev INTEGER REFERENCES feature(id), aspect TEXT, name TEXT NOT NULL, definition TEXT, polarity TEXT, examples TEXT, round INTEGER);
CREATE INDEX IF NOT EXISTS feature_codebook ON feature(codebook);
CREATE TABLE IF NOT EXISTS assignment (
    realization INTEGER NOT NULL REFERENCES realization(id), codebook INTEGER NOT NULL REFERENCES codebook(id),
    feature INTEGER REFERENCES feature(id), confidence TEXT, at TEXT NOT NULL, note TEXT, PRIMARY KEY (realization, codebook));
CREATE TABLE IF NOT EXISTS vector (
    realization INTEGER PRIMARY KEY REFERENCES realization(id), model TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS assignment_feature ON assignment(feature);
CREATE TABLE IF NOT EXISTS flag (
    id INTEGER PRIMARY KEY, codebook INTEGER NOT NULL REFERENCES codebook(id), feature INTEGER NOT NULL REFERENCES feature(id),
    realization INTEGER REFERENCES realization(id), other INTEGER REFERENCES feature(id), verdict TEXT NOT NULL, note TEXT, standing INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS flag_codebook ON flag(codebook);

-- runs of any stage, followed by the GUI
CREATE TABLE IF NOT EXISTS job (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, corpus TEXT, model TEXT, params TEXT, status TEXT NOT NULL,
    total INTEGER, done INTEGER DEFAULT 0, calls INTEGER DEFAULT 0, spent REAL DEFAULT 0, seconds REAL DEFAULT 0,
    recent TEXT, error TEXT, started TEXT, finished TEXT);
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
            self._migrate()

    def _migrate(self) -> None:
        """Columns added after a store was created: provider and billed on call (2026-09-10), realization on reading and head on realization (stage 2)."""
        for table, cols in (("call", (("provider", "TEXT"), ("billed", "REAL"))), ("reading", (("realization", "INTEGER REFERENCES realization(id)"),)), ("realization", (("head", "TEXT"), ("sample", "TEXT"), ("domain_terms", "TEXT"))), ("assignment", (("note", "TEXT"),)), ("feature", (("round", "INTEGER"),)), ("flag", (("standing", "INTEGER DEFAULT 0"),))):
            have = {r[1] for r in self.con.execute(f"PRAGMA table_info({table})")}
            for col, typ in cols:
                if col not in have:
                    self.con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        self.con.commit()                        # called under self.lock from migrate()

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
        sql, params = "SELECT COALESCE(SUM(COALESCE(billed, cost)), 0) FROM call WHERE cached=0", []       # what the server billed when it said, else the list price
        if model:
            sql += " AND model=?"; params.append(model)
        if stage:
            sql += " AND stage=?"; params.append(stage)
        return float(self.rows(sql, params)[0][0])

    def cached_reply(self, model: str, prompt_sha: str) -> Optional[str]:
        r = self.one("SELECT reply FROM call WHERE model=? AND prompt_sha=? AND error IS NULL AND reply IS NOT NULL AND reply != '' ORDER BY id DESC", (model, prompt_sha))
        return r["reply"] if r else None

    def spend_by_model(self) -> list[dict]:
        return [dict(r) for r in self.rows("SELECT model, COUNT(*) n, SUM(cached) cached, SUM(prompt_tokens) prompt_tokens, SUM(completion_tokens) completion_tokens, SUM(COALESCE(billed, cost)) cost, SUM(billed IS NOT NULL) billed_calls FROM call GROUP BY model ORDER BY cost DESC")]


def _plain(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
