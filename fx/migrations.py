"""The store's schema version and the steps between versions.

`PRAGMA user_version` names the version a store is at. Opening a store creates the current schema (CREATE IF NOT EXISTS
for every table) and then runs, in order and each in its own transaction, every step above the store's version; the
store's version is set after each. A store ahead of the code refuses to open. Every step is idempotent: a column is
added only when missing, a fill is INSERT OR IGNORE, a drop is IF EXISTS, so a new store (already at the current
schema) runs them as no-ops.

    fx store version        the store's version and the code's
    fx store migrate        opens the store, which migrates it
"""
from __future__ import annotations

import sqlite3
from typing import Callable


def _add_columns(con: sqlite3.Connection, table: str, cols: tuple[tuple[str, str], ...]) -> None:
    if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
        return
    have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    for col, typ in cols:
        if col not in have:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def step1_columns_and_legacy_copies(con: sqlite3.Connection) -> None:
    """Version 1: the columns stages 1 to 3 added after a store was created, and the legacy tables (assignment, vector,
    fvector) copied into membership and embedding where those lack rows."""
    _add_columns(con, "call", (("provider", "TEXT"), ("billed", "REAL")))
    _add_columns(con, "reading", (("realization", "INTEGER REFERENCES realization(id)"),))
    _add_columns(con, "realization", (("head", "TEXT"), ("sample", "TEXT"), ("domain_terms", "TEXT")))
    _add_columns(con, "assignment", (("note", "TEXT"),))
    _add_columns(con, "feature", (("round", "INTEGER"),))
    _add_columns(con, "flag", (("standing", "INTEGER DEFAULT 0"),))
    con.execute("CREATE INDEX IF NOT EXISTS reading_realization ON reading(realization)")
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for legacy, kind, target, copy in (
            ("assignment", "realization", "membership", "INSERT OR IGNORE INTO membership (kind, unit, codebook, node, confidence, note, at) SELECT 'realization', realization, codebook, feature, confidence, note, at FROM assignment"),
            ("vector", "realization", "embedding", "INSERT OR IGNORE INTO embedding (kind, unit, model, dim, vec) SELECT 'realization', realization, model, dim, vec FROM vector"),
            ("fvector", "feature", "embedding", "INSERT OR IGNORE INTO embedding (kind, unit, model, dim, vec) SELECT 'feature', feature, model, dim, vec FROM fvector")):
        if legacy not in tables:
            continue
        n_legacy = con.execute(f"SELECT COUNT(*) FROM {legacy}").fetchone()[0]
        n_have = con.execute(f"SELECT COUNT(*) FROM {target} WHERE kind=?", (kind,)).fetchone()[0] if n_legacy else 0
        if n_legacy > n_have:
            con.execute(copy)


def step2_imports_as_events(con: sqlite3.Connection) -> None:
    """Version 2: the import table; prompt.import; prompts that arrived before imports were recorded get one synthetic
    import per corpus, dated the corpus's creation, naming the corpus's source."""
    _add_columns(con, "prompt", (("import", "INTEGER REFERENCES import(id)"),))
    for c in con.execute("SELECT c.id, c.source, c.at FROM corpus c WHERE EXISTS (SELECT 1 FROM prompt p WHERE p.corpus=c.id AND p.import IS NULL)").fetchall():
        n = con.execute("SELECT COUNT(*) FROM prompt WHERE corpus=? AND import IS NULL", (c[0],)).fetchone()[0]
        cur = con.execute("INSERT INTO import (corpus, kind, path, domain, added, skipped, unwrapped, at) VALUES (?, 'unrecorded', ?, NULL, ?, 0, 0, ?)", (c[0], c[1], n, c[2]))
        con.execute("UPDATE prompt SET import=? WHERE corpus=? AND import IS NULL", (cur.lastrowid, c[0]))


STEPS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "columns added by stages 1 to 3; legacy tables copied into membership and embedding", step1_columns_and_legacy_copies),
    (2, "imports as events: the import table, prompt.import, one synthetic import per corpus for what was there", step2_imports_as_events),
]
CURRENT = STEPS[-1][0]


def version(con: sqlite3.Connection) -> int:
    return int(con.execute("PRAGMA user_version").fetchone()[0])


class StoreAhead(RuntimeError):
    pass


def run(con: sqlite3.Connection, path: str = "") -> list[int]:
    """Every step above the store's version, in order, each committed with the new version. Returns the steps applied."""
    at = version(con)
    if at > CURRENT:
        raise StoreAhead(f"{path or 'the store'} is at schema version {at}; this code knows up to {CURRENT}: update the code")
    applied = []
    for n, _desc, fn in STEPS:
        if n <= at:
            continue
        con.execute("BEGIN")
        try:
            fn(con)
            con.execute(f"PRAGMA user_version = {n}")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        applied.append(n)
    return applied
