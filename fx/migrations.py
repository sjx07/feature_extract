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


def step3_tags_as_rows(con: sqlite3.Connection) -> None:
    """Version 3: the tag table filled from the domain, system and task columns and from the scalar keys of prompt.meta,
    which lose those keys (provenance, use_case, harvest, file and pasted stay)."""
    import json
    from .tags import NOT_TAGS, promotable
    con.execute("CREATE TABLE IF NOT EXISTS tag (prompt TEXT NOT NULL REFERENCES prompt(id), field TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (prompt, field))")
    con.execute("CREATE INDEX IF NOT EXISTS tag_field ON tag(field, value)")
    for col in ("domain", "system", "task"):
        con.execute(f"INSERT OR IGNORE INTO tag (prompt, field, value) SELECT id, '{col}', {col} FROM prompt WHERE {col} IS NOT NULL AND {col} != ''")
    for r in con.execute("SELECT id, meta FROM prompt WHERE meta IS NOT NULL AND meta != '' AND meta != '{}'").fetchall():
        try:
            meta = json.loads(r[1])
        except ValueError:
            continue
        promoted = promotable(meta)
        if not promoted:
            continue
        for k, v in promoted.items():
            con.execute("INSERT OR IGNORE INTO tag (prompt, field, value) VALUES (?, ?, ?)", (r[0], k, v))
        rest = {k: v for k, v in meta.items() if k not in promoted}
        con.execute("UPDATE prompt SET meta=? WHERE id=?", (json.dumps(rest, ensure_ascii=False), r[0]))


def step4_seed_as_a_scope(con: sqlite3.Connection) -> None:
    """Version 4: codebook.scope ('corpus' | 'seed'), codebook.corpus nullable (the table is rebuilt: SQLite cannot drop a
    NOT NULL), seed codebooks get scope 'seed' and no corpus, and the corpus row named 'seed' goes."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(codebook)")}
    if "scope" not in cols:
        con.execute("""CREATE TABLE codebook_new (
            id INTEGER PRIMARY KEY, corpus INTEGER REFERENCES corpus(id), kind TEXT NOT NULL, version INTEGER NOT NULL,
            model TEXT, round INTEGER NOT NULL, notes TEXT, anchor_agreement REAL, at TEXT NOT NULL, scope TEXT NOT NULL DEFAULT 'corpus')""")
        con.execute("INSERT INTO codebook_new (id, corpus, kind, version, model, round, notes, anchor_agreement, at) SELECT id, corpus, kind, version, model, round, notes, anchor_agreement, at FROM codebook")
        con.execute("DROP INDEX IF EXISTS codebook_version")
        con.execute("DROP TABLE codebook")
        con.execute("ALTER TABLE codebook_new RENAME TO codebook")
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS codebook_version ON codebook(corpus, kind, version)")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS codebook_seed ON codebook(kind, version) WHERE corpus IS NULL")
    seed = con.execute("SELECT id FROM corpus WHERE name='seed'").fetchone()
    if seed:
        con.execute("UPDATE codebook SET scope='seed', corpus=NULL WHERE corpus=?", (seed[0],))
        con.execute("DELETE FROM prompt WHERE corpus=?", (seed[0],))
        con.execute("DELETE FROM import WHERE corpus=?", (seed[0],))
        con.execute("DELETE FROM corpus WHERE id=?", (seed[0],))


def step5_drop_legacy_tables(con: sqlite3.Connection) -> None:
    """Version 5: a last copy of assignment, vector and fvector into membership and embedding, then the four legacy tables
    (alignment too, a pre-refactor branch's, never holding a global) are dropped."""
    step1_columns_and_legacy_copies(con)
    for t in ("assignment", "vector", "fvector", "alignment"):
        con.execute(f"DROP TABLE IF EXISTS {t}")


def step6_checkpoint_schema(con: sqlite3.Connection) -> None:
    """Version 6: a checkpoint records the schema version its blobs were written under; the ones from before are version 1."""
    _add_columns(con, "checkpoint", (("schema", "INTEGER"),))
    con.execute("UPDATE checkpoint SET schema=1 WHERE schema IS NULL")


STEPS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "columns added by stages 1 to 3; legacy tables copied into membership and embedding", step1_columns_and_legacy_copies),
    (2, "imports as events: the import table, prompt.import, one synthetic import per corpus for what was there", step2_imports_as_events),
    (3, "tags as rows: the tag table from the columns and the meta keys", step3_tags_as_rows),
    (4, "the seed as a codebook scope, not a corpus", step4_seed_as_a_scope),
    (5, "the legacy tables assignment, vector, fvector and alignment dropped after a last copy", step5_drop_legacy_tables),
    (6, "checkpoints record their schema version", step6_checkpoint_schema),
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
