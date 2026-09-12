"""The library loop, written once against a Level (see fx.loop.level):

    embed         one vector per unit that has none
    assign        open units onto the tree's nodes or none: a batch against the whole tree, then the open ones
                  against the few nodes nearest by retrieval; a unit the judge removed carries the reason and its
                  node is offered back (putting it back settles the flag)
    judge         read-only: per node, members that do not fit and a split; per group, indistinct siblings; a flag
                  raised again after it was acted on is standing
    reopen        first-time flags send their member open with the reason; anchors never
    candidates    open units that neighbour units from other groups form clusters; the neighbourless are specific
    name          one call per cluster: a variant under a feature, a new feature under a group, or a rejection
    run_round     assign, judge, then reopen -> cluster -> name -> assign -> judge until settled

Nothing written is rewritten: the tree only gains nodes (stamped with their round), a placed unit stays placed unless
the judge's flag reopens it, and only the open units are ever looked at again.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import numpy as np

from ..llm import Client
from ..llm.pool import _sem, run_many
from ..llm.registry import reasoning_low, resolve
from ..store import Store, now
from ..util.ids import parse_id, parse_ids
from ..util.jsonx import extract_object
from .level import Level

MAX_TOKENS = 32768
ASSIGN_SCHEMA = {"type": "object", "properties": {"assignments": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"}, "feature": {"type": ["string", "null"]}, "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}, "required": ["id", "feature", "confidence"]}}}, "required": ["assignments"]}
NAME_SCHEMA = {"type": "object", "properties": {
    "decision": {"type": "string", "enum": ["variant", "feature", "same", "reject"]}, "why": {"type": "string"}, "parent": {"type": ["string", "null"]},
    "group": {"anyOf": [{"type": "string"}, {"type": "null"}, {"type": "object", "properties": {"name": {"type": "string"}, "definition": {"type": "string"}, "aspect": {"type": "string"}}, "required": ["name", "definition", "aspect"], "additionalProperties": False}]},
    "name": {"type": "string"}, "definition": {"type": "string"}, "polarity": {"type": "string", "enum": ["require", "forbid"]},
    "examples": {"type": "array", "items": {"type": "string"}}, "members": {"type": "array", "items": {"type": "string"}}},
    "required": ["decision", "why", "parent", "group", "name", "definition", "polarity", "examples", "members"], "additionalProperties": False}


class Stopped(Exception):
    pass


# ---- vectors
def embed(store: Store, lv: Level, enc=None, model: Optional[str] = None, batch: int = 512) -> dict:
    from ..library.encoders import EMBEDDER, encoder
    model = model or EMBEDDER
    have = {int(r["unit"]) for r in store.rows("SELECT unit FROM embedding WHERE kind=?", (lv.kind,))}
    todo = [u for u in lv.units() if u["id"] not in have]
    if not todo:
        return {"embedded": 0, "model": model}
    enc = enc or encoder(model)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        vecs = enc([u["text"] for u in chunk])
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        with store.lock:
            store.con.executemany("INSERT OR REPLACE INTO embedding (kind, unit, model, dim, vec) VALUES (?,?,?,?,?)",
                                  [(lv.kind, u["id"], model, int(vecs.shape[1]), vecs[k].astype(np.float32).tobytes()) for k, u in enumerate(chunk)])
            store.con.commit()
    return {"embedded": len(todo), "model": model}


def vectors(store: Store, kind: str, ids: list[int]) -> tuple[list[int], np.ndarray]:
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    out_ids, vecs = [], []
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in store.rows(f"SELECT unit, dim, vec FROM embedding WHERE kind=? AND unit IN ({','.join('?' * len(chunk))})", [kind] + chunk):
            out_ids.append(int(r["unit"])); vecs.append(np.frombuffer(r["vec"], dtype=np.float32, count=int(r["dim"])))
    return out_ids, (np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32))


# ---- the tree and membership
def tree(store: Store, lv: Level) -> list[dict]:
    """Groups, features under them, variants under features, each node with support (its own and its children's), its
    anchors' texts and its member count."""
    units = {u["id"]: u for u in lv.units()}
    feats = [dict(r) | {"examples": json.loads(r["examples"] or "[]")} for r in store.rows("SELECT * FROM feature WHERE codebook=? ORDER BY id", (lv.codebook,))]
    mem: dict[int, list[int]] = defaultdict(list)
    for r in store.rows("SELECT unit, node FROM membership WHERE kind=? AND codebook=? AND node IS NOT NULL", (lv.kind, lv.codebook)):
        mem[int(r["node"])].append(int(r["unit"]))
    by_parent: dict = defaultdict(list)
    for f in feats:
        by_parent[f["parent"]].append(f)

    def fill(node):
        ms = [units[u] for u in mem.get(node["id"], []) if u in units]
        node["members_n"], node["own"] = len(ms), sum(u["support"] for u in ms)
        node["support"], node["groups_n"] = node["own"], len(set().union(*(u["groups"] for u in ms)) if ms else set())
        node["anchors"] = [units[e]["label"] for e in node["examples"] if e in units]
    out = []
    for g in by_parent.get(None, []):
        g["features"] = []
        for f in [x for x in by_parent.get(g["id"], []) if x["level"] == "feature"]:
            fill(f); f["variants"] = []
            for v in [x for x in by_parent.get(f["id"], []) if x["level"] == "variant"]:
                fill(v); f["variants"].append(v); f["support"] += v["support"]; f["members_n"] += v["members_n"]
            g["features"].append(f)
        g["support"] = sum(f["support"] for f in g["features"])
        out.append(g)
    return out


def nodes(tr: list[dict]) -> list[dict]:
    return [n for g in tr for f in g["features"] for n in [f] + f.get("variants", [])]


def members(store: Store, lv: Level, node: int, limit: int = 60) -> list[dict]:
    units = {u["id"]: u for u in lv.units()}
    rows = store.rows("SELECT unit, confidence, note FROM membership WHERE kind=? AND codebook=? AND node=?", (lv.kind, lv.codebook, node))
    out = [units[int(r["unit"])] | {"confidence": r["confidence"], "note": r["note"]} for r in rows if int(r["unit"]) in units]
    return sorted(out, key=lambda u: -u["support"])[:limit]


def open_units(store: Store, lv: Level) -> list[dict]:
    have = {int(r["unit"]): r for r in store.rows("SELECT unit, node, note FROM membership WHERE kind=? AND codebook=?", (lv.kind, lv.codebook))}
    return [u | {"note": have[u["id"]]["note"] if u["id"] in have else None} for u in lv.units() if u["id"] not in have or have[u["id"]]["node"] is None]


def anchors(store: Store, lv: Level) -> Optional[float]:
    """Share of the nodes' own example units that membership put back on them; None until placed."""
    hits = total = 0
    placed = {int(r["unit"]): r["node"] for r in store.rows("SELECT unit, node FROM membership WHERE kind=? AND codebook=?", (lv.kind, lv.codebook))}
    for f in store.rows("SELECT id, examples FROM feature WHERE codebook=? AND level IN ('feature','variant')", (lv.codebook,)):
        for e in json.loads(f["examples"] or "[]"):
            if e in placed:
                total += 1; hits += int(placed[e] == f["id"])
    agreement = round(hits / total, 3) if total else None
    with store.lock:
        store.con.execute("UPDATE codebook SET anchor_agreement=? WHERE id=?", (agreement, lv.codebook)); store.con.commit()
    return agreement


def _place(store: Store, lv: Level, unit: int, node: Optional[int], confidence: str, note: Optional[str] = None, keep_note_if_open: bool = True) -> None:
    store.con.execute("INSERT INTO membership (kind, unit, codebook, node, confidence, note, at) VALUES (?,?,?,?,?,?,?) ON CONFLICT(kind, unit, codebook) DO UPDATE SET "
                      "node=excluded.node, confidence=excluded.confidence, at=excluded.at, note=CASE WHEN excluded.node IS NULL AND ? THEN membership.note ELSE excluded.note END",
                      (lv.kind, unit, lv.codebook, node, confidence, note, now(), int(keep_note_if_open)))


def judged(store: Store, lv: Level) -> dict[int, tuple[int, str]]:
    """unit -> (the node the judge removed it from, the reason)."""
    out = {}
    for r in store.rows("SELECT unit, note FROM membership WHERE kind=? AND codebook=? AND node IS NULL AND note LIKE 'reopened:%'", (lv.kind, lv.codebook)):
        body = r["note"].split(":", 1)[1]
        nid, _, why = body.partition("|")
        if nid.isdigit():
            out[int(r["unit"])] = (int(nid), why)
    return out


# ---- assign
def _node_vectors(store: Store, lv: Level, tr: list[dict]) -> tuple[list[int], np.ndarray]:
    ids, vecs = [], []
    for n in nodes(tr):
        src = n["examples"] if lv.node_vector == "anchors" else [int(r["unit"]) for r in store.rows("SELECT unit FROM membership WHERE kind=? AND codebook=? AND node=?", (lv.kind, lv.codebook, n["id"]))]
        got, m = vectors(store, lv.kind, src)
        if got:
            v = m.mean(axis=0); ids.append(n["id"]); vecs.append(v / max(float(np.linalg.norm(v)), 1e-9))
    return ids, (np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32))


def _subtree(tr: list[dict], keep: set[int]) -> list[dict]:
    out = []
    for g in tr:
        fs = []
        for f in g["features"]:
            vs = [v for v in f.get("variants", []) if v["id"] in keep]
            if f["id"] in keep or vs:
                fs.append(dict(f) | {"variants": vs})
        if fs:
            out.append(dict(g) | {"features": fs})
    return out


def assign(store: Store, client: Client, lv: Level, model: str, workers: int = 128, batch: Optional[int] = None, effort: str = "low", only_open: bool = False,
           shortlist: bool = True, progress: Optional[Callable[[int, int, dict], None]] = None, stop: Optional[threading.Event] = None) -> dict:
    """Units not yet placed (or every open unit, with only_open) against the whole tree in shuffled batches, then the open ones
    against their nearest nodes. Written per batch; resumable."""
    import random
    batch = batch or lv.batch
    tr = tree(store, lv)
    valid = {n["id"] for n in nodes(tr)}
    seen = {int(r["unit"]) for r in store.rows("SELECT unit FROM membership WHERE kind=? AND codebook=?" + (" AND node IS NOT NULL" if only_open else ""), (lv.kind, lv.codebook))}
    todo = [u for u in lv.units() if u["id"] not in seen]
    random.Random(lv.codebook).shuffle(todo)
    already_open = 0 if only_open else int(store.one("SELECT COUNT(*) k FROM membership WHERE kind=? AND codebook=? AND node IS NULL", (lv.kind, lv.codebook))["k"])
    summary = {"codebook": lv.codebook, "units": len(todo), "batches": 0, "assigned": 0, "leftover": already_open, "unparsed": 0, "second_pass": 0, "second_pass_assigned": 0, "settled": 0, "stopped": False}
    stop = stop or threading.Event()
    jd = judged(store, lv)
    note = f"{lv.label}:assign"
    if not valid:
        with store.lock:                          # no node yet: record the units as open so the cluster step sees them
            for u in todo:
                _place(store, lv, u["id"], None, "low", None)
            store.con.commit()
        summary["leftover"] += len(todo)
        return summary
    jobs = []
    if todo:
        batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
        summary["batches"] = len(batches)
        jobs = [([u | {"judged": jd.get(u["id"])} for u in b], tr, False) for b in batches]
        _run(store, client, lv, jobs, valid, model, workers, effort, note, summary, progress, stop, jd)
    if shortlist and not summary["stopped"]:
        opened = open_units(store, lv)
        oid, om = vectors(store, lv.kind, [u["id"] for u in opened])
        nid, nm = _node_vectors(store, lv, tr)
        lists: dict[int, list[int]] = {}
        if oid and nid:
            sims = om @ nm.T
            for i, uid in enumerate(oid):
                top = [nid[j] for j in np.argsort(-sims[i])[:lv.shortlist_k]]
                if uid in jd and jd[uid][0] not in top:
                    top.append(jd[uid][0])                # the node the judge removed it from is on the list, with the reason
                lists[uid] = top
        by_id = {u["id"]: u for u in opened}
        by_first: dict[int, list[int]] = defaultdict(list)
        for uid, top in lists.items():
            by_first[top[0]].append(uid)
        jobs2 = []
        for first, uids in by_first.items():
            for i in range(0, len(uids), batch):
                chunk = uids[i:i + batch]
                keep = {n for u in chunk for n in lists[u]}
                jobs2.append(([by_id[u] | {"judged": jd.get(u)} for u in chunk], _subtree(tr, keep), True))
        summary["second_pass"] = len(jobs2)
        if jobs2:
            _run(store, client, lv, jobs2, valid, model, workers, effort, note + ":shortlist", summary, progress, stop, jd)
    summary["anchor_agreement"] = anchors(store, lv)
    return summary


def _run(store, client, lv, jobs, valid, model, workers, effort, note, summary, progress, stop, jd) -> None:
    client.stop = stop
    sem = _sem(resolve(model, client.base_url).base_url, max(workers, 1))
    extra = reasoning_low(model, client.base_url) if effort == "low" else None

    def one(k: int):
        if stop.is_set():
            return k, None
        b, tr, second = jobs[k]
        with sem:
            if stop.is_set():
                return k, None
            return k, client.complete(lv.prompt_assign(tr, b, second), model=model, max_tokens=MAX_TOKENS, extra_body=extra, stage="library", note=note, system=lv.system, schema=ASSIGN_SCHEMA)

    done = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [ex.submit(one, k) for k in range(len(jobs))]
            for fut in as_completed(futs):
                k, r = fut.result()
                if r is None:
                    continue
                if r.error and r.error.startswith("denied"):
                    stop.set(); raise RuntimeError(r.error)
                done += 1
                b, _, second = jobs[k]
                obj = extract_object(r.text, "assignments") if r.text else None
                if obj is None:
                    summary["unparsed"] += 1
                else:
                    got = {}
                    for a in obj["assignments"]:
                        if not isinstance(a, dict):
                            continue
                        uid = parse_id(a.get("id"), {u["id"] for u in b})
                        if uid is None:
                            continue
                        target = a.get("feature", a.get("global"))
                        nid = parse_id(target, valid) if target not in (None, "", "null") else None
                        conf = str(a.get("confidence") or "medium").lower()
                        got[uid] = (nid, conf if conf in ("high", "medium", "low") else "medium")
                    with store.lock:
                        for u in b:
                            nid, conf = got.get(u["id"], (None, "low"))
                            if second and nid is None:
                                continue                                      # the second pass only adds
                            if nid is not None and jd and u["id"] in jd and jd[u["id"]][0] == nid:
                                store.con.execute("UPDATE flag SET standing=1 WHERE codebook=? AND feature=? AND verdict='misfit' AND (realization=? OR other=?)", (lv.codebook, nid, u["id"], u["id"]))
                                summary["settled"] += 1
                            _place(store, lv, u["id"], nid, conf)
                            if second:
                                summary["second_pass_assigned"] += 1; summary["leftover"] -= 1; summary["assigned"] += 1
                            else:
                                summary["assigned" if nid else "leftover"] += 1
                        store.con.commit()
                if progress:
                    progress(done, len(jobs), {"batch": k, "second_pass": second, "unparsed": obj is None, "assigned": summary["assigned"], "leftover": summary["leftover"], "cost": r.cost, "calls": 1})
                if stop.is_set():
                    summary["stopped"] = True
                    ex.shutdown(wait=False, cancel_futures=True); client.close(); break
    finally:
        client.stop = None


# ---- judge, reopen
def judge(store: Store, client: Client, lv: Level, model: str, workers: int = 128, effort: str = "low", progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    tr = tree(store, lv)
    jobs, meta = [], []
    for n in nodes(tr):
        ms = members(store, lv, n["id"])
        if len(ms) >= 2:
            jobs.append(lv.prompt_judge(n, ms, {m["id"]: lv.member_samples(m) for m in ms})); meta.append(("node", n, {m["id"] for m in ms}))
    if lv.prompt_siblings:
        for g in tr:
            if len(g["features"]) >= 2:
                jobs.append(lv.prompt_siblings(g, {f["id"]: members(store, lv, f["id"], 5) for f in g["features"]})); meta.append(("group", g, {f["id"] for f in g["features"]}))
    summary = {"codebook": lv.codebook, "calls": len(jobs), "misfits": 0, "splits": 0, "indistinct": 0, "unparsed": 0, "new": 0, "standing": 0}
    previous = {(int(r["feature"]), r["realization"] and int(r["realization"]), r["other"] and int(r["other"]), r["verdict"]) for r in store.rows("SELECT feature, realization, other, verdict FROM flag WHERE codebook=?", (lv.codebook,))}
    with store.lock:
        store.con.execute("DELETE FROM flag WHERE codebook=?", (lv.codebook,)); store.con.commit()
    if not jobs:
        return summary
    replies = run_many(client, jobs, model=model, workers=workers, max_inflight=workers, stage="library", note=f"{lv.label}:judge", system=lv.system, max_tokens=MAX_TOKENS,
                       extra_body=reasoning_low(model, client.base_url) if effort == "low" else None)
    member_col = "realization" if lv.kind == "realization" else "other"
    for (what, node, valid), r in zip(meta, replies):
        obj = extract_object(r.text) if r and r.text else None
        if obj is None:
            summary["unparsed"] += 1; continue
        rows = []
        if what == "node":
            for m in obj.get("misfits") or []:
                uid = parse_id(m.get("id") if isinstance(m, dict) else m, valid)
                if uid is not None:
                    rows.append({"codebook": lv.codebook, "feature": node["id"], "realization": None, "other": None, "verdict": "misfit", "note": str((m.get("why") if isinstance(m, dict) else "") or "")[:300]} | {member_col: uid})
                    summary["misfits"] += 1
            sp = obj.get("split")
            if isinstance(sp, dict) and len(sp.get("parts") or []) >= 2:
                parts = [{"name": str(p.get("name") or ""), "members": parse_ids(p.get("members"), valid)} for p in sp["parts"] if isinstance(p, dict)]
                rows.append({"codebook": lv.codebook, "feature": node["id"], "realization": None, "other": None, "verdict": "split", "note": json.dumps({"why": str(sp.get("why") or "")[:300], "parts": parts})})
                summary["splits"] += 1
        else:
            for pr in obj.get("indistinct") or []:
                if isinstance(pr, dict):
                    a, b = parse_id(pr.get("a"), valid), parse_id(pr.get("b"), valid)
                    if a is not None and b is not None and a != b:
                        rows.append({"codebook": lv.codebook, "feature": a, "realization": None, "other": b, "verdict": "indistinct", "note": str(pr.get("why") or "")[:300]}); summary["indistinct"] += 1
        for row in rows:
            row["standing"] = int((row["feature"], row["realization"], row["other"], row["verdict"]) in previous)
            summary["standing" if row["standing"] else "new"] += 1
            store.insert("flag", row)
        if progress:
            progress(summary["calls"], len(jobs), {"what": what, "id": node["id"]})
    return summary


def reopen(store: Store, lv: Level) -> dict:
    """First-time misfits and split members go open with the judge's reason; a node's own anchors never; standing flags stay."""
    anchor_set = {(int(f["id"]), int(e)) for f in store.rows("SELECT id, examples FROM feature WHERE codebook=?", (lv.codebook,)) for e in json.loads(f["examples"] or "[]")}
    member_col = "realization" if lv.kind == "realization" else "other"
    todo = []
    for r in store.rows(f"SELECT feature, {member_col} m, note FROM flag WHERE codebook=? AND verdict='misfit' AND standing=0 AND {member_col} IS NOT NULL", (lv.codebook,)):
        if (int(r["feature"]), int(r["m"])) not in anchor_set:
            todo.append((int(r["feature"]), int(r["m"]), (r["note"] or "")[:200], "misfit"))
    for r in store.rows("SELECT feature, note FROM flag WHERE codebook=? AND verdict='split' AND standing=0", (lv.codebook,)):
        n = json.loads(r["note"] or "{}")
        for part in n.get("parts", []):
            for m in part.get("members", []):
                if (int(r["feature"]), int(m)) not in anchor_set:
                    todo.append((int(r["feature"]), int(m), f"the judge sees a distinct sub-feature here ({part.get('name', '')}): {n.get('why', '')}"[:200], "split"))
    counts = {"misfit": 0, "split": 0}
    with store.lock:
        for nid, uid, why, what in todo:
            counts[what] += store.con.execute("UPDATE membership SET node=NULL, confidence='low', note=?, at=? WHERE kind=? AND codebook=? AND unit=? AND node=?",
                                              (f"reopened:{nid}|{why}", now(), lv.kind, lv.codebook, uid, nid)).rowcount
        store.con.commit()
    return {"codebook": lv.codebook, "reopened_misfits": counts["misfit"], "reopened_split_members": counts["split"]}


# ---- candidates, name
def candidates(store: Store, lv: Level, tau: Optional[float] = None) -> dict:
    if tau is None:
        tau = (lv.measure_tau() if lv.measure_tau else None) or lv.tau
    opened = open_units(store, lv)
    ids, m = vectors(store, lv.kind, [u["id"] for u in opened])
    by_id = {u["id"]: u for u in opened}
    parent = list(range(len(ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    has_neighbour = [False] * len(ids)
    if len(ids) > 1:
        sims = m @ m.T
        for i in range(len(ids)):
            for j in np.where(sims[i] >= tau)[0]:
                if j <= i or by_id[ids[i]]["polarity"] != by_id[ids[j]]["polarity"]:
                    continue
                gi, gj = by_id[ids[i]]["groups"], by_id[ids[j]]["groups"]
                if gi & gj == gi | gj:
                    continue                                    # the same group (prompt, or corpus) saying it twice is not support
                has_neighbour[i] = has_neighbour[j] = True
                parent[find(i)] = find(j)
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ids)):
        comps[find(i)].append(i)
    clusters = []
    for mem in comps.values():
        us = [by_id[ids[i]] for i in mem]
        ngroups = len(set().union(*(u["groups"] for u in us)))
        if len(us) >= lv.min_members and ngroups >= lv.min_groups:
            clusters.append({"members": sorted(us, key=lambda u: -u["support"]), "groups": ngroups})
    clusters.sort(key=lambda c: (-c["groups"], -len(c["members"])))
    specific = [ids[i] for i in range(len(ids)) if not has_neighbour[i]]
    with store.lock:
        for u in opened:
            if u["note"] is None and u["id"] not in {int(r["unit"]) for r in []}:
                store.con.execute("INSERT OR IGNORE INTO membership (kind, unit, codebook, node, confidence, note, at) VALUES (?,?,?,NULL,NULL,NULL,?)", (lv.kind, u["id"], lv.codebook, now()))
        store.con.executemany("UPDATE membership SET note='specific' WHERE kind=? AND codebook=? AND unit=? AND node IS NULL AND (note IS NULL OR note='specific')", [(lv.kind, lv.codebook, u) for u in specific])
        store.con.executemany("UPDATE membership SET note=NULL WHERE kind=? AND codebook=? AND unit=? AND note='specific'", [(lv.kind, lv.codebook, u) for u in ids if u not in specific])
        store.con.commit()
    in_cluster = {u["id"] for c in clusters for u in c["members"]}
    return {"tau": round(tau, 3), "open": len(ids), "specific": len(specific), "clusters": clusters, "in_clusters": len(in_cluster), "unclustered": len(ids) - len(specific) - len(in_cluster)}


def name(store: Store, client: Client, lv: Level, clusters: list[dict], round_: int, model: str, workers: int = 16, progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    tr = tree(store, lv)
    features = {n["id"]: n for n in nodes(tr) if n["level"] == "feature"}
    group_ids = {g["id"]: g for g in tr}
    summary = {"codebook": lv.codebook, "round": round_, "clusters": len(clusters), "variants": 0, "features": 0, "groups": 0, "rejected": 0, "unparsed": 0, "assigned": 0}
    if not clusters:
        return summary
    replies = run_many(client, [lv.prompt_name(tr, c["members"]) for c in clusters], model=model, workers=workers, max_inflight=workers, stage="library",
                       note=f"{lv.label}:name:r{round_}", system=lv.system, max_tokens=MAX_TOKENS, schema=NAME_SCHEMA)
    for k, (c, r) in enumerate(zip(clusters, replies)):
        obj = extract_object(r.text) if r and r.text else None
        member_ids = {u["id"] for u in c["members"]}
        by_id = {u["id"]: u for u in c["members"]}
        if not isinstance(obj, dict) or obj.get("decision") not in ("variant", "feature", "same", "reject"):
            summary["unparsed"] += 1; continue
        decision = "feature" if obj["decision"] == "same" else obj["decision"]
        kept = parse_ids(obj.get("members"), member_ids)
        ngroups = len(set().union(*(by_id[u]["groups"] for u in kept))) if kept else 0
        if decision == "reject" or len(kept) < lv.named_min_members or ngroups < lv.named_min_groups or not str(obj.get("name") or "").strip():
            summary["rejected"] += 1
            if progress:
                progress(k + 1, len(clusters), {"cluster": k, "decision": "reject"})
            continue
        pol = str(obj.get("polarity") or "require").lower()
        row = {"codebook": lv.codebook, "prev": None, "aspect": None, "name": str(obj["name"]).strip(), "definition": str(obj.get("definition") or "").strip(),
               "polarity": pol if pol in ("require", "forbid") else "require", "examples": parse_ids(obj.get("examples"), member_ids) or kept[:3], "round": round_}
        parent = parse_id(obj.get("parent"), set(features)) if decision == "variant" and lv.allow_variant else None
        if parent is not None:
            row |= {"level": "variant", "parent": parent}
        else:
            g = obj.get("group")
            gid = parse_id(g, set(group_ids)) if isinstance(g, str) else None
            if gid is None and isinstance(g, dict) and str(g.get("name") or "").strip():
                key = str(g["name"]).strip().lower()
                gid = next((i for i, x in group_ids.items() if x["name"].strip().lower() == key), None)
                if gid is None:
                    aspect = str(g.get("aspect") or "other").lower()
                    gid = store.insert("feature", {"codebook": lv.codebook, "level": "group", "parent": None, "prev": None, "aspect": aspect if aspect in lv.aspects else "other",
                                                   "name": str(g["name"]).strip(), "definition": str(g.get("definition") or "").strip(), "polarity": None, "examples": [], "round": round_})
                    group_ids[gid] = {"id": gid, "name": str(g["name"]).strip()}; summary["groups"] += 1
            if gid is None:
                summary["rejected"] += 1; continue
            row |= {"level": "feature", "parent": gid}
        nid = store.insert("feature", row)
        summary["variants" if row["level"] == "variant" else "features"] += 1
        with store.lock:
            for uid in kept:
                store.con.execute("INSERT INTO membership (kind, unit, codebook, node, confidence, note, at) VALUES (?,?,?,?,'high','named',?) ON CONFLICT(kind, unit, codebook) DO UPDATE SET node=excluded.node, confidence='high', note='named', at=excluded.at WHERE membership.node IS NULL",
                                  (lv.kind, uid, lv.codebook, nid, now()))
            store.con.commit()
        summary["assigned"] += len(kept)
        if progress:
            progress(k + 1, len(clusters), {"cluster": k, "decision": row["level"], "name": row["name"]})
    return summary


# ---- the loop
def run_round(store: Store, client: Client, level, *, batch_model: str, codebook_model: str, workers: int = 128, effort: str = "low", rounds: int = 5,
              tau: Optional[float] = None, min_yield: int = 3, encoder=None, before: Optional[Callable[[Callable], None]] = None,
              log: Optional[Callable[[str, dict], None]] = None, progress=None, stop: Optional[threading.Event] = None) -> dict:
    """assign, judge, then reopen -> cluster -> name -> assign -> judge until settled. `before(step)` lets a level run its own
    first steps (collapse, cold start) through the same step logger; `level` may be a callable, resolved after `before`."""
    stop = stop or threading.Event()
    steps: list[dict] = []

    def step(nm: str, fn) -> dict:
        r = fn()
        steps.append({"step": nm, **{k: v for k, v in r.items() if k not in ("notes", "clusters")}})
        if log:
            log(nm, {k: v for k, v in r.items() if k != "clusters"})
        if r.get("stopped") or stop.is_set():
            raise Stopped()
        return r

    why = f"{rounds} rounds done"
    try:
        if before:
            before(step)
        lv = level() if callable(level) else level
        step("embed", lambda: embed(store, lv, enc=encoder))
        step("assign", lambda: assign(store, client, lv, batch_model, workers=workers, effort=effort, progress=progress, stop=stop))
        step("judge", lambda: judge(store, client, lv, batch_model, workers=workers, effort=effort, progress=progress))
        first = int(store.one("SELECT COALESCE(MAX(round), 0) r FROM feature WHERE codebook=?", (lv.codebook,))["r"]) + 1
        for rnd in range(first, first + rounds):
            r = step("reopen", lambda: reopen(store, lv))
            c = step("cluster", lambda: candidates(store, lv, tau=tau))
            if not c["clusters"] and r["reopened_misfits"] + r["reopened_split_members"] == 0:
                why = f"settled: every flag is standing and no candidate cluster is left ({c['specific']} specific, {c['unclustered']} unclustered open units)"; break
            n = step("name", lambda: name(store, client, lv, c["clusters"], rnd, codebook_model, workers=min(workers, 16), progress=progress)) if c["clusters"] else {"variants": 0, "features": 0, "clusters": 0}
            step("assign", lambda: assign(store, client, lv, batch_model, workers=workers, effort=effort, only_open=True, progress=progress, stop=stop))
            j = step("judge", lambda: judge(store, client, lv, batch_model, workers=workers, effort=effort, progress=progress))
            if n["variants"] + n["features"] < min_yield and j["new"] == 0:
                why = f"settled: round {rnd} named {n['variants'] + n['features']} nodes and the judge raised nothing new ({j['standing']} standing flags)"; break
    except Stopped:
        why = "stopped"
    return {"steps": steps, "stopped_because": why}
