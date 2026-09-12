"""The seed library: a codebook on the corpus named 'seed' whose feature rows are the global features. Reading it gives
the tree (groups, globals) with each global's members per corpus and its support summed over them; a member is a
per-corpus feature, whose own support already includes its variants."""
from __future__ import annotations

import json
from typing import Optional

from ..corpus import get_corpus
from ..store import Store, now
from .cards import SEED, cards


def seed_codebook(store: Store, kind: str) -> int:
    cid = get_corpus(store, SEED, "align")
    r = store.one("SELECT id FROM codebook WHERE corpus=? AND kind=?", (cid, kind))
    if r:
        return int(r["id"])
    return store.insert("codebook", {"corpus": cid, "kind": kind, "version": 1, "model": None, "round": 0, "notes": "the seed library: global features aligned across corpora", "at": now()})


def globals_(store: Store, kind: str) -> list[dict]:
    """[{group..., features: [{global..., members: [card...], corpora: n, support}]}]"""
    cb = seed_codebook(store, kind)
    by_card = {c["id"]: c for c in cards(store, kind)}
    members: dict[int, list[dict]] = {}
    for r in store.rows("SELECT feature, global, confidence, note FROM alignment WHERE global IS NOT NULL"):
        c = by_card.get(int(r["feature"]))
        if c:
            members.setdefault(int(r["global"]), []).append(c | {"confidence": r["confidence"], "note": r["note"]})
    out = []
    for g in store.rows("SELECT * FROM feature WHERE codebook=? AND level='group' ORDER BY id", (cb,)):
        gd = dict(g) | {"features": []}
        for f in store.rows("SELECT * FROM feature WHERE parent=? AND level='feature' ORDER BY id", (g["id"],)):
            ms = sorted(members.get(int(f["id"]), []), key=lambda m: (m["corpus"], -m["support"]))
            gd["features"].append(dict(f) | {"examples": json.loads(f["examples"] or "[]"), "members": ms, "corpora": len({m["corpus"] for m in ms}), "support": sum(m["support"] for m in ms)})
        gd["support"] = sum(f["support"] for f in gd["features"])
        out.append(gd)
    return out


def open_cards(store: Store, kind: str) -> list[dict]:
    """Per-corpus features on no global yet (never aligned, or aligned to NULL), with their note."""
    have = {int(r["feature"]): r for r in store.rows("SELECT feature, global, note FROM alignment")}
    out = []
    for c in cards(store, kind):
        a = have.get(c["id"])
        if a is None or a["global"] is None:
            out.append(c | {"note": a["note"] if a else None})
    return out


def status(store: Store, kind: str) -> dict:
    cs = cards(store, kind)
    have = {int(r["feature"]): r for r in store.rows("SELECT feature, global, note FROM alignment")}
    aligned = [c for c in cs if have.get(c["id"]) and have[c["id"]]["global"] is not None]
    ds = [c for c in cs if have.get(c["id"]) and have[c["id"]]["global"] is None and (have[c["id"]]["note"] or "") == "domain-specific"]
    cb = seed_codebook(store, kind)
    g = store.one("SELECT SUM(level='feature') f, SUM(level='group') g, MAX(round) r FROM feature WHERE codebook=?", (cb,))
    fl = store.one("SELECT COUNT(*) k, COALESCE(SUM(standing),0) s FROM flag WHERE codebook=?", (cb,))
    per_corpus = {}
    for c in cs:
        d = per_corpus.setdefault(c["corpus"], {"features": 0, "aligned": 0, "domain_specific": 0, "support": 0, "aligned_support": 0})
        d["features"] += 1; d["support"] += c["support"]
        if have.get(c["id"]) and have[c["id"]]["global"] is not None:
            d["aligned"] += 1; d["aligned_support"] += c["support"]
        elif c in ds:
            d["domain_specific"] += 1
    return {"kind": kind, "codebook": cb, "cards": len(cs), "aligned": len(aligned), "domain_specific": len(ds), "open": len(cs) - len(aligned) - len(ds),
            "globals": int(g["f"] or 0), "groups": int(g["g"] or 0), "rounds": int(g["r"] or 0), "flags": int(fl["k"]), "standing": int(fl["s"]), "per_corpus": per_corpus}
