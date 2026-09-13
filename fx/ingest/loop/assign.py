"""Step: assign. Open units onto the tree's nodes or none: shuffled batches against the whole tree, then the open ones
against the few nodes nearest by retrieval. A unit the judge removed carries the reason and its node is offered back;
putting it back settles the flag."""
from __future__ import annotations

import numpy as np
import threading
from fx.core.llm import Client
from fx.core.llm.pool import _sem
from fx.core.llm.registry import reasoning_low, resolve
from fx.core.store import Store
from fx.core.util.ids import parse_id
from fx.core.util.jsonx import extract_object
from fx.ingest.loop.level import Level
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional
from fx.ingest.loop.calls import ASSIGN_SCHEMA, MAX_TOKENS
from fx.ingest.loop.state import node_vectors, _place, anchors, judged, nodes, open_units, tree, vectors

# ---- assign
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
        nid, nm = node_vectors(store, lv, tr)
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
    if not summary["stopped"] and jd:
        # the assigner has adjudicated every reopened unit it was offered: the ones it left open drop the judge's reason,
        # so they are ordinary open units from here (at the feature level, a seed once, then specific)
        offered = {u["id"] for b, _, _ in jobs for u in b} | ({u["id"] for b, _, _ in jobs2 for u in b} if shortlist else set())
        with store.lock:
            store.con.executemany("UPDATE membership SET note=NULL WHERE kind=? AND codebook=? AND unit=? AND node IS NULL AND note LIKE 'reopened:%'", [(lv.kind, lv.codebook, u) for u in offered if u in jd])
            store.con.commit()
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
