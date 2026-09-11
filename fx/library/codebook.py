"""A codebook is a version of the library: the feature tree (group rows, feature rows with parent = their group), what
assignment put on each feature, the judge's flags, and the anchor agreement. This module reads and writes that; the
step modules (coldstart, assign, judge, revise) decide what goes in it.

Settings shared by the steps live here too: the cold-start model, the minimum support of a feature, the batch size.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from ..corpus import corpus_id
from ..store import Store, now
from ..util.ids import parse_id, parse_ids
from . import prompts as P

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


def groups(store: Store, cb: int) -> list[dict]:
    """The codebook tree with each feature's support under this version: readings and prompts, from assignments."""
    feats = [dict(r) | {"examples": json.loads(r["examples"] or "[]")} for r in store.rows("SELECT * FROM feature WHERE codebook=? ORDER BY id", (cb,))]
    sup = {int(r["feature"]): (int(r["n"]), int(r["prompts"]), int(r["k"])) for r in
           store.rows("SELECT a.feature, SUM(r.n) n, SUM(r.prompts) prompts, COUNT(*) k FROM assignment a JOIN realization r ON r.id=a.realization WHERE a.codebook=? AND a.feature IS NOT NULL GROUP BY a.feature", (cb,))}
    ex_ids = sorted({e for f in feats for e in f["examples"]})
    ex_text = {int(r["id"]): r["declaration"] for r in store.rows(f"SELECT id, declaration FROM realization WHERE id IN ({','.join('?' * len(ex_ids)) or 'NULL'})", ex_ids)} if ex_ids else {}
    out = []
    for g in [f for f in feats if f["level"] == "group"]:
        g["features"] = []
        for f in [f for f in feats if f["level"] == "feature" and f["parent"] == g["id"]]:
            f["readings"], f["support"], f["realizations"] = sup.get(f["id"], (0, 0, 0))
            f["anchors"] = [ex_text[e] for e in f["examples"] if e in ex_text]
            g["features"].append(f)
        g["support"] = sum(f["support"] for f in g["features"])
        out.append(g)
    return out


def new_codebook(store: Store, cid: int, kind: str, model: str, round_: int, notes: str = "") -> tuple[int, int]:
    r = store.one("SELECT MAX(version) v FROM codebook WHERE corpus=? AND kind=?", (cid, kind))
    version = int(r["v"] or 0) + 1
    return store.insert("codebook", {"corpus": cid, "kind": kind, "version": version, "model": model, "round": round_, "notes": notes, "at": now()}), version


def write_codebook(store: Store, cb: int, obj: dict, kind: str, realization_ids: set[int], old: Optional[list[dict]] = None) -> dict:
    """Write the groups and features of a reply into codebook `cb`. With `old` (the previous version's tree), an id the
    reply kept links the new row to its predecessor through prev; an unknown id is treated as new."""
    old_groups = {g["id"]: g for g in (old or [])}
    old_feats = {f["id"]: f for g in (old or []) for f in g["features"]}
    aspects = P.ASPECTS_GUIDANCE if kind == "guidance" else P.ASPECTS_MATERIAL
    n_groups = n_feats = kept = 0
    for g in obj.get("groups") or []:
        if not isinstance(g, dict) or not str(g.get("name") or "").strip():
            continue
        gprev = parse_id(g.get("id"), set(old_groups)) if old else None
        aspect = str(g.get("aspect") or "other").lower()
        gid = store.insert("feature", {"codebook": cb, "level": "group", "parent": None, "prev": gprev, "aspect": aspect if aspect in aspects else "other",
                                       "name": str(g["name"]).strip(), "definition": str(g.get("definition") or "").strip(), "polarity": None, "examples": []})
        n_groups += 1
        for f in g.get("features") or []:
            if not isinstance(f, dict) or not str(f.get("name") or "").strip():
                continue
            fprev = parse_id(f.get("id"), set(old_feats)) if old else None
            kept += fprev is not None
            pol = str(f.get("polarity") or "require").lower()
            ex = parse_ids(f.get("examples"), realization_ids) or (parse_ids(f.get("replaces"), realization_ids) if not old else [])   # a cold start that misfiled its examples
            if not ex and fprev is not None:
                ex = old_feats[fprev]["examples"]
            store.insert("feature", {"codebook": cb, "level": "feature", "parent": gid, "prev": fprev, "aspect": None, "name": str(f["name"]).strip(),
                                     "definition": str(f.get("definition") or "").strip(), "polarity": pol if pol in ("require", "forbid") else "require", "examples": ex})
            n_feats += 1
    return {"groups": n_groups, "features": n_feats, "kept": kept}


def _codebook_schema(revise: bool) -> dict:
    """The reply shape of COLDSTART (no ids) or REVISE (ids kept, replaces, retired). Closed objects, so a key the prompt
    did not ask for cannot appear; the pilot's cold start had put the example ids under "replaces" when both were allowed."""
    feat = {"name": {"type": "string"}, "definition": {"type": "string"}, "polarity": {"type": "string", "enum": ["require", "forbid"]},
            "examples": {"type": "array", "items": {"type": "string"}}}
    grp = {"name": {"type": "string"}, "definition": {"type": "string"}, "aspect": {"type": "string"}}
    top = {}
    if revise:
        feat = {"id": {"type": ["string", "null"]}, **feat, "replaces": {"type": "array", "items": {"type": "string"}}}
        grp = {"id": {"type": ["string", "null"]}, **grp}
        top = {"notes": {"type": "string"}, "retired": {"type": "array", "items": {"type": "string"}}}
    fschema = {"type": "object", "properties": feat, "required": list(feat), "additionalProperties": False}
    gschema = {"type": "object", "properties": {**grp, "features": {"type": "array", "items": fschema}}, "required": list(grp) + ["features"], "additionalProperties": False}
    return {"type": "object", "properties": {**top, "groups": {"type": "array", "items": gschema}}, "required": list(top) + ["groups"], "additionalProperties": False}


COLDSTART_SCHEMA, REVISE_SCHEMA = _codebook_schema(False), _codebook_schema(True)
ASSIGN_SCHEMA = {"type": "object", "properties": {"assignments": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"}, "feature": {"type": ["string", "null"]}, "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}, "required": ["id", "feature", "confidence"]}}}, "required": ["assignments"]}


def anchors(store: Store, cb: int) -> Optional[float]:
    """Share of the features' own example declarations that assignment put back on them; None until assigned."""
    hits = total = 0
    for f in store.rows("SELECT id, examples FROM feature WHERE codebook=? AND level='feature'", (cb,)):
        for rid in json.loads(f["examples"] or "[]"):
            a = store.one("SELECT feature FROM assignment WHERE codebook=? AND realization=?", (cb, rid))
            if a is None:
                continue
            total += 1
            hits += int(a["feature"] == f["id"])
    agreement = round(hits / total, 3) if total else None
    with store.lock:
        store.con.execute("UPDATE codebook SET anchor_agreement=? WHERE id=?", (agreement, cb)); store.con.commit()
    return agreement


def members(store: Store, cb: int, fid: int, limit: int = 60) -> list[dict]:
    return [dict(r) | {"conditions": json.loads(r["conditions"] or "[]")} for r in
            store.rows("SELECT r.*, a.confidence FROM assignment a JOIN realization r ON r.id=a.realization WHERE a.codebook=? AND a.feature=? ORDER BY r.prompts DESC, r.n DESC LIMIT ?", (cb, fid, limit))]


def flags(store: Store, cb: int) -> list[dict]:
    return [dict(r) for r in store.rows("SELECT f.*, x.name feature_name, y.name other_name, r.declaration FROM flag f JOIN feature x ON x.id=f.feature LEFT JOIN feature y ON y.id=f.other "
                                        "LEFT JOIN realization r ON r.id=f.realization WHERE f.codebook=? ORDER BY f.verdict, f.feature", (cb,))]


def leftovers(store: Store, cb: int, corpus: str, kind: str, low: bool = True) -> list[dict]:
    conf = "OR a.confidence='low'" if low else ""
    return [dict(r) | {"conditions": json.loads(r["conditions"] or "[]")} for r in
            store.rows(f"SELECT r.*, a.confidence FROM assignment a JOIN realization r ON r.id=a.realization WHERE a.codebook=? AND (a.feature IS NULL {conf}) ORDER BY r.prompts DESC, r.n DESC", (cb,))]


def status(store: Store, corpus: str, kind: str) -> dict:
    cid = corpus_id(store, corpus)
    rz = store.one("SELECT COUNT(*) k, COALESCE(SUM(n),0) n, COALESCE(SUM(prompts),0) p FROM realization WHERE corpus=? AND kind=?", (cid, kind))
    versions = []
    for c in store.rows("SELECT * FROM codebook WHERE corpus=? AND kind=? ORDER BY version", (cid, kind)):
        cb = int(c["id"])
        f = store.one("SELECT SUM(level='feature') f, SUM(level='group') g FROM feature WHERE codebook=?", (cb,))
        a = store.one("SELECT COUNT(*) k, SUM(feature IS NOT NULL) assigned, SUM(feature IS NULL) leftover, SUM(confidence='low') low FROM assignment WHERE codebook=?", (cb,))
        cov = store.one("SELECT COALESCE(SUM(CASE WHEN a.feature IS NOT NULL THEN r.n ELSE 0 END),0) n FROM assignment a JOIN realization r ON r.id=a.realization WHERE a.codebook=?", (cb,))
        fl = store.one("SELECT COUNT(*) k FROM flag WHERE codebook=?", (cb,))
        versions.append(dict(c) | {"features": int(f["f"] or 0), "groups": int(f["g"] or 0), "assigned": int(a["assigned"] or 0), "leftover": int(a["leftover"] or 0), "low": int(a["low"] or 0),
                                   "unassigned": int(rz["k"]) - int(a["k"] or 0), "reading_coverage": round(int(cov["n"]) / int(rz["n"]), 3) if rz["n"] else None, "flags": int(fl["k"])})
    return {"corpus": corpus, "kind": kind, "realizations": int(rz["k"]), "readings": int(rz["n"]), "versions": versions}
