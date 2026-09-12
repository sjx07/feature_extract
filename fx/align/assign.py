"""Open cards onto the global features, or none: each card against the few globals nearest to it by retrieval (a global's
vector is the mean of its members'), in batches that share a nearest global. A card the judge removed from a global
carries the reason, and that global is offered back: the assigner adjudicates, and putting it back settles the flag."""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Callable, Optional

import numpy as np

from ..llm import Client
from ..llm.pool import run_many
from ..llm.registry import DEFAULT_MODEL, reasoning_low
from ..store import Store, now
from ..util.ids import parse_id
from ..util.jsonx import extract_object
from . import prompts as P
from .embed import vectors
from .seed import globals_, open_cards

MAX_TOKENS = 32768
BATCH = 12
SCHEMA = {"type": "object", "properties": {"alignments": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"}, "global": {"type": ["string", "null"]}, "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}, "required": ["id", "global", "confidence"]}}}, "required": ["alignments"]}


def global_vectors(store: Store, tree: list[dict]) -> tuple[list[int], np.ndarray]:
    ids, vecs = [], []
    for g in tree:
        for s in g["features"]:
            mids = [m["id"] for m in s["members"]]
            got, m = vectors(store, mids)
            if got:
                v = m.mean(axis=0); ids.append(s["id"]); vecs.append(v / max(float(np.linalg.norm(v)), 1e-9))
    return ids, (np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32))


def judged(store: Store) -> dict[int, tuple[int, str]]:
    out = {}
    for r in store.rows("SELECT feature, note FROM alignment WHERE global IS NULL AND note LIKE 'reopened:S%'"):
        sid, _, why = r["note"].split(":S", 1)[1].partition("|")
        out[int(r["feature"])] = (int(sid), why)
    return out


def assign(store: Store, client: Client, kind: str, model: str = DEFAULT_MODEL, workers: int = 64, batch: int = BATCH, k: int = 5, effort: str = "low",
           progress: Optional[Callable[[int, int, dict], None]] = None, stop: Optional[threading.Event] = None) -> dict:
    tree = globals_(store, kind)
    summary = {"kind": kind, "open": 0, "batches": 0, "aligned": 0, "left_open": 0, "settled": 0, "unparsed": 0}
    opened = open_cards(store, kind)
    summary["open"] = len(opened)
    sids, sm = global_vectors(store, tree)
    if not opened or not sids:
        # nothing to align against yet: record the open cards so the cluster step sees them
        with store.lock:
            for c in opened:
                store.con.execute("INSERT OR IGNORE INTO alignment (feature, global, confidence, note, at) VALUES (?,NULL,NULL,NULL,?)", (c["id"], now()))
            store.con.commit()
        return summary
    oid, om = vectors(store, [c["id"] for c in opened])
    by_id = {c["id"]: c for c in opened}
    jd = judged(store)
    sims = om @ sm.T
    lists: dict[int, list[int]] = {}
    for i, fid in enumerate(oid):
        top = [sids[j] for j in np.argsort(-sims[i])[:k]]
        if fid in jd and jd[fid][0] not in top:
            top.append(jd[fid][0])
        lists[fid] = top
    by_first: dict[int, list[int]] = defaultdict(list)
    for fid, top in lists.items():
        by_first[top[0]].append(fid)
    globals_by_id = {s["id"]: s for g in tree for s in g["features"]}
    jobs = []
    for first, fids in by_first.items():
        for i in range(0, len(fids), batch):
            chunk = fids[i:i + batch]
            keep = {s for f in chunk for s in lists[f]}
            sub = [dict(g) | {"features": [s for s in g["features"] if s["id"] in keep]} for g in tree]
            sub = [g for g in sub if g["features"]]
            cards_ = [by_id[f] | ({"judged": jd[f]} if f in jd else {}) for f in chunk]
            jobs.append((cards_, sub))
    summary["batches"] = len(jobs)
    prompts = []
    for cards_, sub in jobs:
        text = P.assign(sub, cards_, shortlist=True)
        for c in cards_:
            if c.get("judged"):
                text = text.replace(f"F{c['id']} [", f"F{c['id']} (the judge removed this from S{c['judged'][0]}: {c['judged'][1][:160]}; put it back only if the judge is wrong) [", 1)
        prompts.append(text)
    replies = run_many(client, prompts, model=model, workers=workers, max_inflight=workers, stage="align", note=f"align:{kind}:assign", system=P.SYSTEM, max_tokens=MAX_TOKENS,
                       schema=SCHEMA, extra_body=reasoning_low(model, client.base_url) if effort == "low" else None)
    valid = set(globals_by_id)
    for n, ((cards_, _), r) in enumerate(zip(jobs, replies)):
        obj = extract_object(r.text, "alignments") if r and r.text else None
        if obj is None:
            summary["unparsed"] += 1
            continue
        got = {}
        for a in obj["alignments"]:
            if isinstance(a, dict):
                fid = parse_id(a.get("id"), {c["id"] for c in cards_})
                if fid is not None:
                    got[fid] = (parse_id(a.get("global"), valid) if a.get("global") not in (None, "", "null") else None, str(a.get("confidence") or "medium"))
        with store.lock:
            for c in cards_:
                sid, conf = got.get(c["id"], (None, "low"))
                if sid is not None and c["id"] in jd and jd[c["id"]][0] == sid:
                    store.con.execute("UPDATE flag SET standing=1 WHERE feature=? AND other=? AND verdict='misfit'", (sid, c["id"])); summary["settled"] += 1
                store.con.execute("INSERT INTO alignment (feature, global, confidence, note, at) VALUES (?,?,?,NULL,?) ON CONFLICT(feature) DO UPDATE SET global=excluded.global, confidence=excluded.confidence, at=excluded.at, "
                                  "note=CASE WHEN excluded.global IS NULL THEN alignment.note ELSE NULL END", (c["id"], sid, conf, now()))
                summary["aligned" if sid is not None else "left_open"] += 1
            store.con.commit()
        if progress:
            progress(n + 1, len(jobs), {"batch": n, "aligned": summary["aligned"], "left_open": summary["left_open"], "cost": r.cost, "calls": 1})
    return summary
