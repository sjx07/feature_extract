"""The seed library: a codebook on the corpus named 'seed' whose feature rows are the global features. Reading it gives
the tree (groups, globals) with each global's members per corpus and its support summed over them; a member is a
per-corpus feature, whose own support already includes its variants."""
from __future__ import annotations

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
    """The seed tree with each global's members (cards) per corpus; see fx.loop.engine.tree for the rest."""
    from ..loop import tree
    from .level import feature_level
    lv = feature_level(store, kind)
    by_card = {c["id"]: c for c in cards(store, kind)}
    mem: dict[int, list[dict]] = {}
    for r in store.rows("SELECT unit, node, confidence, note FROM membership WHERE kind='feature' AND codebook=? AND node IS NOT NULL", (lv.codebook,)):
        c = by_card.get(int(r["unit"]))
        if c:
            mem.setdefault(int(r["node"]), []).append(c | {"confidence": r["confidence"], "note": r["note"]})
    out = tree(store, lv)
    for g in out:
        for f in g["features"]:
            f["members"] = sorted(mem.get(f["id"], []), key=lambda m: (m["corpus"], -m["support"]))
            f["corpora"] = f["groups_n"]
    return out


def open_cards(store: Store, kind: str) -> list[dict]:
    from ..loop import open_units
    from .level import feature_level
    return open_units(store, feature_level(store, kind))


def status(store: Store, kind: str) -> dict:
    cb = seed_codebook(store, kind)
    cs = cards(store, kind)
    have = {int(r["unit"]): r for r in store.rows("SELECT unit, node, note FROM membership WHERE kind='feature' AND codebook=?", (cb,))}
    aligned = [c for c in cs if have.get(c["id"]) and have[c["id"]]["node"] is not None]
    ds = [c for c in cs if have.get(c["id"]) and have[c["id"]]["node"] is None and (have[c["id"]]["note"] or "") == "specific"]
    g = store.one("SELECT SUM(level='feature') f, SUM(level='group') g, MAX(round) r FROM feature WHERE codebook=?", (cb,))
    fl = store.one("SELECT COUNT(*) k, COALESCE(SUM(standing),0) s FROM flag WHERE codebook=?", (cb,))
    per_corpus = {}
    for c in cs:
        d = per_corpus.setdefault(c["corpus"], {"features": 0, "aligned": 0, "domain_specific": 0, "support": 0, "aligned_support": 0})
        d["features"] += 1; d["support"] += c["support"]
        if c in aligned:
            d["aligned"] += 1; d["aligned_support"] += c["support"]
        elif c in ds:
            d["domain_specific"] += 1
    return {"kind": kind, "codebook": cb, "cards": len(cs), "aligned": len(aligned), "domain_specific": len(ds), "open": len(cs) - len(aligned) - len(ds),
            "globals": int(g["f"] or 0), "groups": int(g["g"] or 0), "rounds": int(g["r"] or 0), "flags": int(fl["k"]), "standing": int(fl["s"]), "per_corpus": per_corpus}
