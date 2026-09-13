"""A codebook is a corpus's library of one kind: the tree (group rows, feature rows under them, variant rows under
features), what assignment put on each node, the judge's flags, and the anchor agreement. The tree only grows. This
module reads and writes it; the step modules (coldstart, assign, judge, name) decide what goes in it.

Settings shared by the steps live here too: the cold-start model, the minimum support of a feature, the batch size.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from fx.data.corpus import corpus_id
from fx.core.store import Store, now
from fx.core.util.ids import parse_ids
from fx.ingest.induce import prompts as P

COLDSTART_MODEL = "gpt-5.6-sol"      # one whole-corpus call per kind; the assign and judge batches use the decomposition model
MIN_SUPPORT = 3
BATCH = 20
MAX_TOKENS = 32768
CONTEXT_TOKENS = int(os.environ.get("FX_CONTEXT_TOKENS", 120_000))    # what one cold-start or revise prompt may carry; declarations past it wait for the loop


def fit(rows: list[dict], budget_tokens: int = CONTEXT_TOKENS, overhead_tokens: int = 3000) -> tuple[list[dict], int]:
    """The head of a support-ordered declaration list that fits the budget (4 chars a token, 12 tokens of ids and counts a line).
    Returns (kept, dropped)."""
    left = (budget_tokens - overhead_tokens) * 4
    out = []
    for r in rows:
        cost = len(r["declaration"]) + 24 + sum(len(c) for c in (r.get("conditions") or [])[:2])
        if cost > left:
            break
        out.append(r); left -= cost
    return out, len(rows) - len(out)
KINDS = ("guidance", "material")


def latest(store: Store, corpus: str, kind: str, version: Optional[int] = None) -> Optional[dict]:
    cid = corpus_id(store, corpus)
    if version is None:
        r = store.one("SELECT * FROM codebook WHERE corpus=? AND kind=? ORDER BY version DESC", (cid, kind))
    else:
        r = store.one("SELECT * FROM codebook WHERE corpus=? AND kind=? AND version=?", (cid, kind, version))
    return dict(r) if r else None


def new_codebook(store: Store, cid: int, kind: str, model: str, round_: int, notes: str = "") -> tuple[int, int]:
    r = store.one("SELECT MAX(version) v FROM codebook WHERE corpus=? AND kind=?", (cid, kind))
    version = int(r["v"] or 0) + 1
    return store.insert("codebook", {"corpus": cid, "kind": kind, "version": version, "model": model, "round": round_, "notes": notes, "at": now()}), version


def write_codebook(store: Store, cb: int, obj: dict, kind: str, realization_ids: set[int], round_: int = 0) -> dict:
    """Write the groups and features of a cold-start reply into codebook `cb`."""
    aspects = P.ASPECTS_GUIDANCE if kind == "guidance" else P.ASPECTS_MATERIAL
    n_groups = n_feats = 0
    for g in obj.get("groups") or []:
        if not isinstance(g, dict) or not str(g.get("name") or "").strip():
            continue
        aspect = str(g.get("aspect") or "other").lower()
        gid = store.insert("feature", {"codebook": cb, "level": "group", "parent": None, "prev": None, "aspect": aspect if aspect in aspects else "other",
                                       "name": str(g["name"]).strip(), "definition": str(g.get("definition") or "").strip(), "polarity": None, "examples": [], "round": round_})
        n_groups += 1
        for f in g.get("features") or []:
            if not isinstance(f, dict) or not str(f.get("name") or "").strip():
                continue
            pol = str(f.get("polarity") or "require").lower()
            ex = parse_ids(f.get("examples"), realization_ids) or parse_ids(f.get("replaces"), realization_ids)   # a cold start that misfiled its examples
            store.insert("feature", {"codebook": cb, "level": "feature", "parent": gid, "prev": None, "aspect": None, "name": str(f["name"]).strip(),
                                     "definition": str(f.get("definition") or "").strip(), "polarity": pol if pol in ("require", "forbid") else "require", "examples": ex, "round": round_})
            n_feats += 1
    return {"groups": n_groups, "features": n_feats}


def _codebook_schema() -> dict:
    """The COLDSTART reply shape, closed objects."""
    feat = {"name": {"type": "string"}, "definition": {"type": "string"}, "polarity": {"type": "string", "enum": ["require", "forbid"]},
            "examples": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5}}     # a feature without a cited wording has no anchor; a fresh text2cypher listing came back with none
    grp = {"name": {"type": "string"}, "definition": {"type": "string"}, "aspect": {"type": "string"}}
    fschema = {"type": "object", "properties": feat, "required": list(feat), "additionalProperties": False}
    gschema = {"type": "object", "properties": {**grp, "features": {"type": "array", "items": fschema}}, "required": list(grp) + ["features"], "additionalProperties": False}
    return {"type": "object", "properties": {"groups": {"type": "array", "items": gschema}}, "required": ["groups"], "additionalProperties": False}


COLDSTART_SCHEMA = _codebook_schema()
def anchors(store: Store, cb: int) -> Optional[float]:
    from fx.ingest.loop import anchors as _anchors
    from fx.ingest.induce.level import wording_level
    return _anchors(store, wording_level(store, cb))


def groups(store: Store, cb: int) -> list[dict]:
    """The codebook tree with support from membership; see fx.ingest.loop.tree."""
    from fx.ingest.loop import tree
    from fx.ingest.induce.level import wording_level
    return tree(store, wording_level(store, cb))


def nodes(tr: list[dict]) -> list[dict]:
    return [n for g in tr for f in g["features"] for n in [f] + f.get("variants", [])]


def members(store: Store, cb: int, fid: int, limit: int = 60) -> list[dict]:
    from fx.ingest.loop import members as _members
    from fx.ingest.induce.level import wording_level
    return _members(store, wording_level(store, cb), fid, limit)


def flags(store: Store, cb: int) -> list[dict]:
    return [dict(r) for r in store.rows("SELECT f.*, x.name feature_name, y.name other_name, r.declaration FROM flag f JOIN feature x ON x.id=f.feature LEFT JOIN feature y ON y.id=f.other "
                                        "LEFT JOIN realization r ON r.id=f.realization WHERE f.codebook=? ORDER BY f.verdict, f.feature", (cb,))]


def leftovers(store: Store, cb: int, corpus: str, kind: str, low: bool = True) -> list[dict]:
    conf = "OR m.confidence='low'" if low else ""
    return [dict(r) | {"conditions": json.loads(r["conditions"] or "[]")} for r in
            store.rows(f"SELECT r.*, m.confidence, m.note FROM membership m JOIN realization r ON r.id=m.unit WHERE m.kind='realization' AND m.codebook=? AND (m.node IS NULL {conf}) ORDER BY r.prompts DESC, r.n DESC", (cb,))]


def status(store: Store, corpus: str, kind: str) -> dict:
    cid = corpus_id(store, corpus)
    rz = store.one("SELECT COUNT(*) k, COALESCE(SUM(n),0) n, COALESCE(SUM(prompts),0) p FROM realization WHERE corpus=? AND kind=?", (cid, kind))
    versions = []
    for c in store.rows("SELECT * FROM codebook WHERE corpus=? AND kind=? ORDER BY version", (cid, kind)):
        cb = int(c["id"])
        f = store.one("SELECT SUM(level='feature') f, SUM(level='group') g, SUM(level='variant') v, MAX(round) rounds FROM feature WHERE codebook=?", (cb,))
        a = store.one("SELECT COUNT(*) k, SUM(node IS NOT NULL) assigned, SUM(node IS NULL) leftover, SUM(confidence='low') low, SUM(note='specific') specific, SUM(note='named') named FROM membership WHERE kind='realization' AND codebook=?", (cb,))
        cov = store.one("SELECT COALESCE(SUM(CASE WHEN m.node IS NOT NULL THEN r.n ELSE 0 END),0) n FROM membership m JOIN realization r ON r.id=m.unit WHERE m.kind='realization' AND m.codebook=?", (cb,))
        fl = store.one("SELECT COUNT(*) k, COALESCE(SUM(standing),0) s FROM flag WHERE codebook=?", (cb,))
        versions.append(dict(c) | {"features": int(f["f"] or 0), "groups": int(f["g"] or 0), "variants": int(f["v"] or 0), "rounds": int(f["rounds"] or 0), "specific": int(a["specific"] or 0), "named": int(a["named"] or 0),
                                   "assigned": int(a["assigned"] or 0), "leftover": int(a["leftover"] or 0), "low": int(a["low"] or 0),
                                   "unassigned": int(rz["k"]) - int(a["k"] or 0), "reading_coverage": round(int(cov["n"]) / int(rz["n"]), 3) if rz["n"] else None, "flags": int(fl["k"]), "standing": int(fl["s"])})
    return {"corpus": corpus, "kind": kind, "realizations": int(rz["k"]), "readings": int(rz["n"]), "versions": versions}
