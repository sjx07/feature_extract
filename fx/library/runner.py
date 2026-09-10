"""Stage 2, the feature library of a corpus, one per kind (guidance, material). Five steps over the store:

    collapse(store, corpus, kind)                       readings -> realizations (distinct declarations), no calls
    coldstart(store, client, corpus, kind, model)       one call over every declaration -> codebook version 1
    assign(store, client, corpus, kind, model)          batches of declarations on the latest version -> assignments
    judge(store, client, corpus, kind, model)           read-only coherence: misfits, splits, indistinct siblings -> flags
    revise(store, client, corpus, kind, model)          leftovers + flags -> the next version (the only writer after cold start)

A version is a full snapshot; assign runs from scratch on each version, so no assignment depends on the path that
produced it. The anchor agreement of a version is the share of its features' own example declarations that assign
puts back on them; it is measured, stored on the codebook row, and shown, and a revision that drops it is the
thing to read first.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import defaultdict
from typing import Callable, Optional

from ..llm import Client
from ..llm.pool import run_many
from ..llm.registry import DEFAULT_MODEL, price, resolve
from ..store import Store, now
from ..util.jsonx import extract_object
from . import prompts as P

COLDSTART_MODEL = "gpt-5.6-sol"      # one whole-corpus call per kind; the assign and judge batches use the decomposition model
MIN_SUPPORT = 3
BATCH = 40
MAX_TOKENS = 32768
KINDS = ("guidance", "material")
_ID = re.compile(r"^[RFG]?(\d+)$")


class Stop(Exception):
    pass


# ---- readings -> realizations
def _key(polarity: str, declaration: str) -> str:
    d = re.sub(r"\s+", " ", (declaration or "").strip().lower()).strip(" .;:,")
    return f"{polarity or 'require'}|{d}"


def corpus_id(store: Store, corpus: str) -> int:
    r = store.one("SELECT id FROM corpus WHERE name=?", (corpus,))
    if not r:
        raise KeyError(f"no corpus {corpus!r}")
    return int(r["id"])


def collapse(store: Store, corpus: str, kind: str) -> dict:
    """Group the corpus's readings of one kind by (polarity, normalised declaration); write realizations and point each
    reading at its realization. Idempotent: an existing realization keeps its id, its counts are refreshed."""
    cid = corpus_id(store, corpus)
    span_kind = "atom" if kind == "guidance" else "material"
    rows = store.rows("SELECT r.id, r.prompt, r.polarity, r.declaration, r.condition FROM reading r JOIN span s ON s.id=r.span JOIN prompt p ON p.id=r.prompt "
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
            vals = (best["polarity"] or "require", best["declaration"], len(rs), len({r["prompt"] for r in rs}), json.dumps(conds))
            if key in have:
                rid = have[key]
                con.execute("UPDATE realization SET polarity=?, declaration=?, n=?, prompts=?, conditions=? WHERE id=?", vals + (rid,))
            else:
                rid = con.execute("INSERT INTO realization (corpus, kind, key, polarity, declaration, n, prompts, conditions) VALUES (?,?,?,?,?,?,?,?)", (cid, kind, key) + vals).lastrowid
                have[key] = rid; new += 1
            con.executemany("UPDATE reading SET realization=? WHERE id=?", [(rid, r["id"]) for r in rs])
        con.commit()
    return {"corpus": corpus, "kind": kind, "readings": len(rows), "realizations": len(groups), "new": new}


def realizations(store: Store, corpus: str, kind: str) -> list[dict]:
    cid = corpus_id(store, corpus)
    return [dict(r) | {"conditions": json.loads(r["conditions"] or "[]")} for r in
            store.rows("SELECT * FROM realization WHERE corpus=? AND kind=? ORDER BY prompts DESC, n DESC, id", (cid, kind))]


# ---- codebooks
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
    out = []
    for g in [f for f in feats if f["level"] == "group"]:
        g["features"] = []
        for f in [f for f in feats if f["level"] == "feature" and f["parent"] == g["id"]]:
            f["readings"], f["support"], f["realizations"] = sup.get(f["id"], (0, 0, 0))
            g["features"].append(f)
        g["support"] = sum(f["support"] for f in g["features"])
        out.append(g)
    return out


def _new_codebook(store: Store, cid: int, kind: str, model: str, round_: int, notes: str = "") -> tuple[int, int]:
    r = store.one("SELECT MAX(version) v FROM codebook WHERE corpus=? AND kind=?", (cid, kind))
    version = int(r["v"] or 0) + 1
    return store.insert("codebook", {"corpus": cid, "kind": kind, "version": version, "model": model, "round": round_, "notes": notes, "at": now()}), version


def _ids(xs, valid: set[int]) -> list[int]:
    out = []
    for x in xs if isinstance(xs, list) else []:
        m = _ID.match(str(x).strip())
        if m and int(m.group(1)) in valid:
            out.append(int(m.group(1)))
    return out


def _id(x, valid: set[int]) -> Optional[int]:
    got = _ids([x], valid)
    return got[0] if got else None


def _call(client: Client, prompt: str, model: str, note: str, schema=None) -> str:
    r = client.complete(prompt, model=model, max_tokens=MAX_TOKENS, stage="library", note=note, system=P.SYSTEM, schema=schema)
    if r.error and r.error.startswith("denied"):
        raise RuntimeError(r.error)
    return r.text


def _write_codebook(store: Store, cb: int, obj: dict, kind: str, realization_ids: set[int], old: Optional[list[dict]] = None) -> dict:
    """Write the groups and features of a reply into codebook `cb`. With `old` (the previous version's tree), an id the
    reply kept links the new row to its predecessor through prev; an unknown id is treated as new."""
    old_groups = {g["id"]: g for g in (old or [])}
    old_feats = {f["id"]: f for g in (old or []) for f in g["features"]}
    aspects = P.ASPECTS_GUIDANCE if kind == "guidance" else P.ASPECTS_MATERIAL
    n_groups = n_feats = kept = 0
    for g in obj.get("groups") or []:
        if not isinstance(g, dict) or not str(g.get("name") or "").strip():
            continue
        gprev = _id(g.get("id"), set(old_groups)) if old else None
        aspect = str(g.get("aspect") or "other").lower()
        gid = store.insert("feature", {"codebook": cb, "level": "group", "parent": None, "prev": gprev, "aspect": aspect if aspect in aspects else "other",
                                       "name": str(g["name"]).strip(), "definition": str(g.get("definition") or "").strip(), "polarity": None, "examples": []})
        n_groups += 1
        for f in g.get("features") or []:
            if not isinstance(f, dict) or not str(f.get("name") or "").strip():
                continue
            fprev = _id(f.get("id"), set(old_feats)) if old else None
            kept += fprev is not None
            pol = str(f.get("polarity") or "require").lower()
            ex = _ids(f.get("examples"), realization_ids) or (_ids(f.get("replaces"), realization_ids) if not old else [])   # a cold start that misfiled its examples
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


def coldstart(store: Store, client: Client, corpus: str, kind: str, model: str = COLDSTART_MODEL, min_support: int = MIN_SUPPORT, domain: Optional[str] = None) -> dict:
    cid = corpus_id(store, corpus)
    decl = realizations(store, corpus, kind)
    if not decl:
        raise ValueError(f"no {kind} realizations for {corpus}: run collapse, or decompose first")
    domain = domain or (store.one("SELECT domain FROM prompt WHERE corpus=? AND domain IS NOT NULL", (cid,)) or {"domain": corpus})["domain"] or corpus
    reply = _call(client, P.coldstart(kind, domain, decl, min_support), model, f"{corpus}:{kind}:coldstart", schema=COLDSTART_SCHEMA)
    obj = extract_object(reply, "groups")
    if obj is None:
        raise RuntimeError("cold start reply was not a codebook (no 'groups' list)")
    cb, version = _new_codebook(store, cid, kind, model, 0)
    w = _write_codebook(store, cb, obj, kind, {d["id"] for d in decl})
    return {"corpus": corpus, "kind": kind, "codebook": cb, "version": version, "declarations": len(decl), **w}


def assign(store: Store, client: Client, corpus: str, kind: str, model: str = DEFAULT_MODEL, version: Optional[int] = None, workers: int = 16, batch: int = BATCH,
           progress: Optional[Callable[[int, int, dict], None]] = None, stop: Optional[threading.Event] = None) -> dict:
    """Every realization not yet assigned under the version, in batches; a batch whose reply does not parse is left for the next run."""
    cbrow = latest(store, corpus, kind, version)
    if not cbrow:
        raise ValueError(f"no codebook for {corpus} {kind}: run coldstart first")
    cb = int(cbrow["id"])
    tree = groups(store, cb)
    valid = {f["id"] for g in tree for f in g["features"]}
    done_ids = {int(r["realization"]) for r in store.rows("SELECT realization FROM assignment WHERE codebook=?", (cb,))}
    todo = [d for d in realizations(store, corpus, kind) if d["id"] not in done_ids]
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    summary = {"codebook": cb, "version": cbrow["version"], "realizations": len(todo), "batches": len(batches), "assigned": 0, "leftover": 0, "unparsed": 0, "stopped": False}
    if not batches:
        anchors(store, cb)
        return summary
    stop = stop or threading.Event()
    client.stop = stop
    prompts_ = [P.assign(kind, tree, b) for b in batches]
    done = 0

    def on_progress(k, total):
        pass
    try:
        replies = run_many(client, prompts_, model=model, workers=workers, max_inflight=workers, stage="library", note=f"{corpus}:{kind}:assign:v{cbrow['version']}",
                           system=P.SYSTEM, max_tokens=MAX_TOKENS, schema=ASSIGN_SCHEMA, progress=on_progress)
    finally:
        client.stop = None
    for b, r in zip(batches, replies):
        done += 1
        obj = extract_object(r.text, "assignments") if r and r.text else None
        if obj is None:
            summary["unparsed"] += 1
            if progress:
                progress(done, len(batches), {"batch": done, "unparsed": True})
            continue
        got = {}
        for a in obj["assignments"]:
            if not isinstance(a, dict):
                continue
            rid = _id(a.get("id"), {d["id"] for d in b})
            if rid is None:
                continue
            fid = _id(a.get("feature"), valid) if a.get("feature") not in (None, "", "null") else None
            conf = str(a.get("confidence") or "medium").lower()
            got[rid] = (fid, conf if conf in ("high", "medium", "low") else "medium")
        with store.lock:
            for d in b:
                fid, conf = got.get(d["id"], (None, "low"))     # a declaration the reply skipped is a leftover at low confidence
                store.con.execute("INSERT OR REPLACE INTO assignment (realization, codebook, feature, confidence, at) VALUES (?,?,?,?,?)", (d["id"], cb, fid, conf, now()))
                summary["assigned" if fid else "leftover"] += 1
            store.con.commit()
        if progress:
            progress(done, len(batches), {"batch": done, "assigned": summary["assigned"], "leftover": summary["leftover"]})
        if stop.is_set():
            summary["stopped"] = True
            break
    summary["anchor_agreement"] = anchors(store, cb)
    return summary


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


def judge(store: Store, client: Client, corpus: str, kind: str, model: str = DEFAULT_MODEL, version: Optional[int] = None, workers: int = 16,
          progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    """Read-only coherence on one version: misfit members and splits per feature, indistinct pairs per group. Replaces the version's flags."""
    cbrow = latest(store, corpus, kind, version)
    if not cbrow:
        raise ValueError("no codebook")
    cb = int(cbrow["id"])
    tree = groups(store, cb)
    feats = [f for g in tree for f in g["features"]]
    jobs, meta = [], []
    for f in feats:
        ms = members(store, cb, f["id"])
        if len(ms) >= 2:
            jobs.append(P.judge_members(f, ms)); meta.append(("feature", f, {m["id"] for m in ms}))
    for g in tree:
        if len(g["features"]) >= 2:
            samples = {f["id"]: members(store, cb, f["id"], 5) for f in g["features"]}
            jobs.append(P.judge_siblings(g, samples)); meta.append(("group", g, {f["id"] for f in g["features"]}))
    replies = run_many(client, jobs, model=model, workers=workers, max_inflight=workers, stage="library", note=f"{corpus}:{kind}:judge:v{cbrow['version']}", system=P.SYSTEM, max_tokens=MAX_TOKENS) if jobs else []
    summary = {"codebook": cb, "version": cbrow["version"], "calls": len(jobs), "misfits": 0, "splits": 0, "indistinct": 0, "unparsed": 0}
    with store.lock:
        store.con.execute("DELETE FROM flag WHERE codebook=?", (cb,)); store.con.commit()
    for (what, node, valid), r in zip(meta, replies):
        obj = extract_object(r.text) if r and r.text else None
        if obj is None:
            summary["unparsed"] += 1
            continue
        rows = []
        if what == "feature":
            for m in obj.get("misfits") or []:
                rid = _id(m.get("id") if isinstance(m, dict) else m, valid)
                if rid is not None:
                    rows.append({"codebook": cb, "feature": node["id"], "realization": rid, "other": None, "verdict": "misfit", "note": str((m.get("why") if isinstance(m, dict) else "") or "")[:300]})
            sp = obj.get("split")
            if isinstance(sp, dict) and len(sp.get("parts") or []) >= 2:
                parts = [{"name": str(p.get("name") or ""), "members": _ids(p.get("members"), valid)} for p in sp["parts"] if isinstance(p, dict)]
                rows.append({"codebook": cb, "feature": node["id"], "realization": None, "other": None, "verdict": "split", "note": json.dumps({"why": str(sp.get("why") or "")[:300], "parts": parts})})
                summary["splits"] += 1
            summary["misfits"] += sum(1 for x in rows if x["verdict"] == "misfit")
        else:
            for pr in obj.get("indistinct") or []:
                if not isinstance(pr, dict):
                    continue
                a, b = _id(pr.get("a"), valid), _id(pr.get("b"), valid)
                if a is not None and b is not None and a != b:
                    rows.append({"codebook": cb, "feature": a, "realization": None, "other": b, "verdict": "indistinct", "note": str(pr.get("why") or "")[:300]})
                    summary["indistinct"] += 1
        for row in rows:
            store.insert("flag", row)
        if progress:
            progress(summary["calls"], len(jobs), {"what": what, "id": node["id"]})
    return summary


def flags(store: Store, cb: int) -> list[dict]:
    return [dict(r) for r in store.rows("SELECT f.*, x.name feature_name, y.name other_name, r.declaration FROM flag f JOIN feature x ON x.id=f.feature LEFT JOIN feature y ON y.id=f.other "
                                        "LEFT JOIN realization r ON r.id=f.realization WHERE f.codebook=? ORDER BY f.verdict, f.feature", (cb,))]


def leftovers(store: Store, cb: int, corpus: str, kind: str, low: bool = True) -> list[dict]:
    conf = "OR a.confidence='low'" if low else ""
    return [dict(r) | {"conditions": json.loads(r["conditions"] or "[]")} for r in
            store.rows(f"SELECT r.*, a.confidence FROM assignment a JOIN realization r ON r.id=a.realization WHERE a.codebook=? AND (a.feature IS NULL {conf}) ORDER BY r.prompts DESC, r.n DESC", (cb,))]


def revise(store: Store, client: Client, corpus: str, kind: str, model: str = COLDSTART_MODEL, version: Optional[int] = None, min_support: int = MIN_SUPPORT,
           max_leftover: int = 600, domain: Optional[str] = None) -> dict:
    """The next version from the current one, its leftovers and its flags, in one call. Assign it afresh afterwards."""
    cbrow = latest(store, corpus, kind, version)
    if not cbrow:
        raise ValueError("no codebook")
    cb, cid = int(cbrow["id"]), int(cbrow["corpus"])
    tree = groups(store, cb)
    left = leftovers(store, cb, corpus, kind)[:max_leftover]
    fl = []
    for x in flags(store, cb):
        if x["verdict"] == "misfit":
            fl.append(f"misfit in F{x['feature']} {x['feature_name']}: R{x['realization']} \"{x['declaration']}\" ({x['note']})")
        elif x["verdict"] == "split":
            n = json.loads(x["note"] or "{}")
            fl.append(f"split suggested for F{x['feature']} {x['feature_name']}: {n.get('why', '')}; parts " + "; ".join(f"{p['name']} [{', '.join('R%d' % m for m in p['members'])}]" for p in n.get("parts", [])))
        else:
            fl.append(f"indistinct: F{x['feature']} {x['feature_name']} and F{x['other']} {x['other_name']} ({x['note']})")
    domain = domain or (store.one("SELECT domain FROM prompt WHERE corpus=? AND domain IS NOT NULL", (cid,)) or {"domain": corpus})["domain"] or corpus
    reply = _call(client, P.revise(kind, domain, tree, left, fl, min_support), model, f"{corpus}:{kind}:revise:v{cbrow['version']}", schema=REVISE_SCHEMA)
    obj = extract_object(reply, "groups")
    if obj is None:
        raise RuntimeError("revise reply was not a codebook (no 'groups' list)")
    all_r = {d["id"] for d in realizations(store, corpus, kind)}
    ncb, nversion = _new_codebook(store, cid, kind, model, int(cbrow["round"]) + 1, str(obj.get("notes") or "")[:2000])
    w = _write_codebook(store, ncb, obj, kind, all_r, old=tree)
    old_feats = {f["id"] for g in tree for f in g["features"]}
    retired = _ids(obj.get("retired"), old_feats)
    return {"corpus": corpus, "kind": kind, "codebook": ncb, "version": nversion, "leftover_seen": len(left), "flags_seen": len(fl), "retired": len(retired), **w, "notes": str(obj.get("notes") or "")[:500]}


# ---- status and preview
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


def preview(store: Store, corpus: str, kind: str, step: str, model: Optional[str] = None, batch: int = BATCH) -> dict:
    """Calls and dollars for one step, from the store's counts and the registry's prices."""
    cid = corpus_id(store, corpus)
    n = int(store.one("SELECT COUNT(*) k FROM realization WHERE corpus=? AND kind=?", (cid, kind))["k"])
    chars = int(store.one("SELECT COALESCE(SUM(LENGTH(declaration)),0) c FROM realization WHERE corpus=? AND kind=?", (cid, kind))["c"])
    cbrow = latest(store, corpus, kind)
    nf = int(store.one("SELECT COUNT(*) k FROM feature WHERE codebook=? AND level='feature'", (cbrow["id"],))["k"]) if cbrow else 40
    ng = int(store.one("SELECT COUNT(*) k FROM feature WHERE codebook=? AND level='group'", (cbrow["id"],))["k"]) if cbrow else 8
    if step in ("coldstart", "revise"):
        model = model or COLDSTART_MODEL
        calls, tin, tout = 1, 1200 + (chars + 30 * n) // 3, 300 + 120 * max(nf, 30)
    elif step == "assign":
        model = model or DEFAULT_MODEL
        done = int(store.one("SELECT COUNT(*) k FROM assignment WHERE codebook=?", (cbrow["id"],))["k"]) if cbrow else 0
        calls = math.ceil(max(n - done, 0) / batch)
        tin, tout = 900 + 40 * nf + 25 * batch, 20 * batch + 2000
    elif step == "judge":
        model = model or DEFAULT_MODEL
        calls, tin, tout = nf + ng, 1500, 2500
    else:
        raise ValueError(step)
    ep = resolve(model)
    pi, po = price(model, ep)
    return {"step": step, "model": model, "endpoint": ep.name, "calls": calls, "tokens_in": calls * tin, "tokens_out": calls * tout, "dollars": round(calls * (tin * pi + tout * po) / 1e6, 3), "realizations": n}
