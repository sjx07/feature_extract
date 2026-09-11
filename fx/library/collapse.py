"""Readings -> realizations: one row per distinct declaration (polarity + normalised wording) of a corpus and kind, with
its support; every reading points at its realization. No model call. Assignment works on realizations, so a wording
that recurs is judged once and support is a join."""
from __future__ import annotations

import json
import re
from collections import defaultdict

from ..corpus import corpus_id
from ..store import Store


def _key(polarity: str, declaration: str) -> str:
    d = re.sub(r"\s+", " ", (declaration or "").strip().lower()).strip(" .;:,")
    return f"{polarity or 'require'}|{d}"


def collapse(store: Store, corpus: str, kind: str) -> dict:
    """Group the corpus's readings of one kind by (polarity, normalised declaration); write realizations and point each
    reading at its realization. Idempotent: an existing realization keeps its id, its counts are refreshed."""
    cid = corpus_id(store, corpus)
    span_kind = "atom" if kind == "guidance" else "material"
    rows = store.rows("SELECT r.id, r.prompt, r.polarity, r.declaration, r.condition, r.verb, s.note FROM reading r JOIN span s ON s.id=r.span JOIN prompt p ON p.id=r.prompt "
                      "WHERE p.corpus=? AND s.kind=? AND r.declaration IS NOT NULL AND r.declaration != ''", (cid, span_kind))
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[_key(r["polarity"], r["declaration"])].append(r)
    have = {r["key"]: int(r["id"]) for r in store.rows("SELECT id, key FROM realization WHERE corpus=? AND kind=?", (cid, kind))}
    new = 0
    with store.lock:
        con = store.con
        for key, rs in groups.items():
            best = max(rs, key=lambda r: len(r["declaration"]))     # the fullest wording stands for the group
            conds = sorted({(r["condition"] or "always") for r in rs}, key=lambda c: (c == "always", c))[:6]
            # the head the cold start groups by: the verb of a guidance reading, the material kind of a material one
            head = (best["note"] or "other") if kind == "material" else re.sub(r"\s+", " ", (best["verb"] or "").strip().lower())
            vals = (best["polarity"] or "require", best["declaration"], len(rs), len({r["prompt"] for r in rs}), json.dumps(conds), head)
            if key in have:
                rid = have[key]
                con.execute("UPDATE realization SET polarity=?, declaration=?, n=?, prompts=?, conditions=?, head=? WHERE id=?", vals + (rid,))
            else:
                rid = con.execute("INSERT INTO realization (corpus, kind, key, polarity, declaration, n, prompts, conditions, head) VALUES (?,?,?,?,?,?,?,?,?)", (cid, kind, key) + vals).lastrowid
                have[key] = rid; new += 1
            con.executemany("UPDATE reading SET realization=? WHERE id=?", [(rid, r["id"]) for r in rs])
        con.commit()
    return {"corpus": corpus, "kind": kind, "readings": len(rows), "realizations": len(groups), "new": new}


def realizations(store: Store, corpus: str, kind: str) -> list[dict]:
    cid = corpus_id(store, corpus)
    return [dict(r) | {"conditions": json.loads(r["conditions"] or "[]")} for r in
            store.rows("SELECT * FROM realization WHERE corpus=? AND kind=? ORDER BY prompts DESC, n DESC, id", (cid, kind))]
