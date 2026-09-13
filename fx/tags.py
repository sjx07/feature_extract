"""A prompt's tags: one row per field in `tag`. The fields are data: a FACET jsonl brings role, family, stage and
subtask; a folder brings none; a user adds any. Three fields, domain, system and task, are also columns on prompt (a
cache the decomposition prompts and the prompt page read); `set_tags` keeps them in step.

    set_tags(store, pid, {"role": "staged"})        upsert; an empty value removes the row
    fields(store)                                   the fields in use, with their value counts
"""
from __future__ import annotations

import json
from typing import Iterable, Optional

from .store import Store

COLUMNS = ("domain", "system", "task")                      # tag fields mirrored as prompt columns
PREFERRED = ("corpus", "domain", "task", "role", "family", "stage", "system", "collection", "bank_source", "subtask")   # the facet order for known fields
NOT_TAGS = {"pasted", "file", "provenance", "use_case", "harvest"}        # meta keys that stay in meta


def promotable(meta: dict) -> dict[str, str]:
    """The meta keys that are tags: top-level scalars, not in NOT_TAGS, non-empty."""
    return {k: str(v) for k, v in (meta or {}).items() if k not in NOT_TAGS and isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v) != ""}


def set_tags(store: Store, pid: str, tags: dict, con=None) -> int:
    """Write the tags of one prompt; a None or empty value deletes the row. Uses the store's connection under its lock unless
    a connection already under the lock is given."""
    def _do(c):
        n = 0
        for field, value in tags.items():
            field = str(field).strip()
            if not field:
                continue
            if value is None or str(value).strip() == "":
                c.execute("DELETE FROM tag WHERE prompt=? AND field=?", (pid, field))
            else:
                c.execute("INSERT INTO tag (prompt, field, value) VALUES (?, ?, ?) ON CONFLICT(prompt, field) DO UPDATE SET value=excluded.value", (pid, field, str(value).strip()))
            if field in COLUMNS:
                c.execute(f"UPDATE prompt SET {field}=? WHERE id=?", (str(value).strip() if value is not None and str(value).strip() else None, pid))
            n += 1
        return n
    if con is not None:
        return _do(con)
    with store.lock:
        n = _do(store.con); store.con.commit()
    return n


def tags_of(store: Store, pids: Iterable[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    ids = list(pids)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        for r in store.rows(f"SELECT prompt, field, value FROM tag WHERE prompt IN ({','.join('?' * len(chunk))})", chunk):
            out.setdefault(r["prompt"], {})[r["field"]] = r["value"]
    return out


def fields(store: Store, corpus: Optional[str] = None) -> list[dict]:
    """The fields in use: name, distinct values, prompts tagged; for one corpus when named."""
    if corpus:
        rows = store.rows("SELECT t.field, COUNT(DISTINCT t.value) k, COUNT(*) n FROM tag t JOIN prompt p ON p.id=t.prompt JOIN corpus c ON c.id=p.corpus WHERE c.name=? GROUP BY t.field", (corpus,))
    else:
        rows = store.rows("SELECT field, COUNT(DISTINCT value) k, COUNT(*) n FROM tag GROUP BY field")
    out = [{"field": r["field"], "values": int(r["k"]), "prompts": int(r["n"])} for r in rows]
    return sorted(out, key=lambda d: (PREFERRED.index(d["field"]) if d["field"] in PREFERRED else 99, d["field"]))


def field_order(names: Iterable[str]) -> list[str]:
    return sorted(set(names), key=lambda f: (PREFERRED.index(f) if f in PREFERRED else 99, f))
