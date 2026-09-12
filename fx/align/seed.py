"""The seed library: a codebook on the corpus named 'seed' whose feature rows are the global features. Reading it gives
the tree (groups, globals) with each global's members per corpus and its support summed over them; a member is a
per-corpus feature, whose own support already includes its variants."""
from __future__ import annotations

from ..corpus import get_corpus
from ..store import Store, now
from .cards import SEED, cards


ASPECT_GROUPS = {
    "guidance": [("role", "who the model is told to be"), ("task", "what the model is asked to do"), ("reasoning", "how the model is told to think or work"),
                 ("answer", "what the answer must contain or omit"), ("format", "the shape and serialisation of the answer"), ("tool", "how tools and the environment are used"),
                 ("safety", "what the model must refuse, avoid or guard"), ("other", "guidance outside the aspects above")],
    "material": [("example", "worked examples"), ("schema", "schemas and data definitions"), ("code", "code the prompt supplies"), ("slot", "input placeholders"),
                 ("template", "output shapes to copy"), ("title", "headings and separators"), ("reference", "pasted documents and facts"), ("other", "other supplied material")],
}


def seed_codebook(store: Store, kind: str) -> int:
    """The seed codebook, born with its groups: one per aspect. A cross-domain library's grouping is the aspect list every
    corpus library already carries, so the naming call picks a group and never invents one."""
    cid = get_corpus(store, SEED, "align")
    r = store.one("SELECT id FROM codebook WHERE corpus=? AND kind=?", (cid, kind))
    if r:
        return int(r["id"])
    cb = store.insert("codebook", {"corpus": cid, "kind": kind, "version": 1, "model": None, "round": 0, "notes": "the seed library: global features aligned across corpora", "at": now()})
    for aspect, definition in ASPECT_GROUPS.get(kind, ASPECT_GROUPS["guidance"]):
        store.insert("feature", {"codebook": cb, "level": "group", "parent": None, "prev": None, "aspect": aspect, "name": aspect, "definition": definition, "polarity": None, "examples": [], "round": 0})
    return cb


def reset(store: Store, kind: str) -> dict:
    """Delete the seed of a kind: its globals and groups, memberships, flags, alignments and codebook row. The cards'
    embeddings stay (they key on the feature, not the seed), so the next round is born fresh without re-embedding."""
    cid = get_corpus(store, SEED, "align")
    r = store.one("SELECT id FROM codebook WHERE corpus=? AND kind=?", (cid, kind))
    if not r:
        return {"codebook": None, "deleted": {}}
    cb = int(r["id"])
    with store.lock:
        d = {"flags": store.con.execute("DELETE FROM flag WHERE codebook=?", (cb,)).rowcount,
             "memberships": store.con.execute("DELETE FROM membership WHERE kind='feature' AND codebook=?", (cb,)).rowcount,
             "alignments": store.con.execute("DELETE FROM alignment WHERE global IN (SELECT id FROM feature WHERE codebook=?)", (cb,)).rowcount,
             "nodes": store.con.execute("DELETE FROM feature WHERE codebook=?", (cb,)).rowcount}
        store.con.execute("DELETE FROM codebook WHERE id=?", (cb,))
        store.con.commit()
    return {"codebook": cb, "deleted": d}


def regroup_by_aspect(store: Store, kind: str) -> dict:
    """Repair for a seed whose naming calls invented groups: move every global under the aspect group its own group named,
    create the aspect groups if missing, drop the groups left empty."""
    cb = seed_codebook(store, kind)
    have = {r["aspect"]: int(r["id"]) for r in store.rows("SELECT id, aspect FROM feature WHERE codebook=? AND level='group' AND name=aspect", (cb,))}
    for aspect, definition in ASPECT_GROUPS.get(kind, ASPECT_GROUPS["guidance"]):
        if aspect not in have:
            have[aspect] = store.insert("feature", {"codebook": cb, "level": "group", "parent": None, "prev": None, "aspect": aspect, "name": aspect, "definition": definition, "polarity": None, "examples": [], "round": 0})
    moved = 0
    with store.lock:
        for g in store.con.execute("SELECT id, aspect FROM feature WHERE codebook=? AND level='group' AND name != aspect", (cb,)).fetchall():
            target = have.get(g["aspect"] or "other", have["other"])
            moved += store.con.execute("UPDATE feature SET parent=? WHERE parent=? AND level='feature'", (target, g["id"])).rowcount
        dropped = store.con.execute("DELETE FROM feature WHERE codebook=? AND level='group' AND name != aspect AND id NOT IN (SELECT DISTINCT parent FROM feature WHERE parent IS NOT NULL)", (cb,)).rowcount
        store.con.commit()
    return {"codebook": cb, "moved": moved, "groups_dropped": dropped, "groups": len(have)}


def globals_(store: Store, kind: str) -> list[dict]:
    """The seed tree with each global's members (cards) per corpus; see fx.loop.tree for the rest."""
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
