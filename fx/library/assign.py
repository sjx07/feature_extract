"""Every realization onto a feature of one codebook version, or onto none, in batches. Resumable: a realization already
assigned under the version is skipped, a batch whose reply does not parse is left for the next run. Runs from scratch
per version, so no assignment depends on the path that produced it. Ends by measuring the version's anchor agreement.

A second pass follows over the leftovers: each wording sees only the features that share its head (the verb of a
feature's name or of its anchors), with their anchors, in batches by head. A choice among a few is easier than among
seventy, so the pass recovers wordings the first pass returned as none (the pilot's first pass missed a quarter of
the features' own anchors).

Each batch is written the moment its reply arrives, so a stopped or killed run keeps what it had. Batches are small
and the model's reasoning effort is low by default: a 40-item batch with the reasoning left at the provider default
spent 25k hidden tokens and 20 minutes per call on DeepSeek v4 flash and hit the 32k ceiling on 10 of 17 calls
(pilot, 2026-09-10); FACET's classification ran at batch 15 and low effort for the same reason."""
from __future__ import annotations

import random
import threading
from typing import Callable, Optional

from ..llm import Client
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..llm.pool import _sem
from ..llm.registry import DEFAULT_MODEL, reasoning_low, resolve
from ..store import Store, now
from ..util.ids import parse_id
from ..util.jsonx import extract_object
from . import prompts as P
from .codebook import ASSIGN_SCHEMA, BATCH, MAX_TOKENS, anchors, groups, latest
from collections import defaultdict
from .collapse import realizations


def _feature_heads(tree: list[dict]) -> dict[str, set[int]]:
    """head -> feature ids whose name or anchors start with that verb."""
    heads: dict[str, set[int]] = defaultdict(set)
    for g in tree:
        for f in g["features"]:
            for text in [f["name"]] + list(f.get("anchors") or []):
                w = (text or "").strip().lower().split()
                if w:
                    heads[w[0]].add(f["id"])
    return heads


def _subtree(tree: list[dict], keep: set[int]) -> list[dict]:
    out = []
    for g in tree:
        fs = [f for f in g["features"] if f["id"] in keep]
        if fs:
            out.append(dict(g) | {"features": fs})
    return out


def assign(store: Store, client: Client, corpus: str, kind: str, model: str = DEFAULT_MODEL, version: Optional[int] = None, workers: int = 128, batch: int = BATCH,
           effort: str = "low", shortlist: bool = True, progress: Optional[Callable[[int, int, dict], None]] = None, stop: Optional[threading.Event] = None) -> dict:
    cbrow = latest(store, corpus, kind, version)
    if not cbrow:
        raise ValueError(f"no codebook for {corpus} {kind}: run coldstart first")
    cb = int(cbrow["id"])
    tree = groups(store, cb)
    valid = {f["id"] for g in tree for f in g["features"]}
    done_ids = {int(r["realization"]) for r in store.rows("SELECT realization FROM assignment WHERE codebook=?", (cb,))}
    todo = [d for d in realizations(store, corpus, kind) if d["id"] not in done_ids]
    # realizations come support-first and, among equals, in prompt order, so a batch would be one prompt's atoms in
    # sequence and the assigner reads the procedure instead of each line; a fixed shuffle per version breaks that
    random.Random(cb).shuffle(todo)
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    summary = {"codebook": cb, "version": cbrow["version"], "realizations": len(todo), "batches": len(batches), "assigned": 0, "leftover": 0, "unparsed": 0,
               "second_pass": 0, "second_pass_assigned": 0, "stopped": False}
    stop = stop or threading.Event()
    note = f"{corpus}:{kind}:assign:v{cbrow['version']}"
    jobs = [(b, tree, False) for b in batches]
    if jobs:
        _run(store, client, cb, kind, jobs, valid, model, workers, effort, note, summary, progress, stop)
    if shortlist and not summary["stopped"]:
        # the second pass: the version's leftovers, each against the features sharing its head, in batches by head
        heads = _feature_heads(tree)
        left = [d for d in realizations(store, corpus, kind) if d["id"] in {int(r["realization"]) for r in store.rows("SELECT realization FROM assignment WHERE codebook=? AND feature IS NULL", (cb,))}]
        by_head: dict[str, list[dict]] = defaultdict(list)
        for d in left:
            if heads.get(d.get("head") or ""):
                by_head[d["head"]].append(d)
        jobs2 = [(hb[i:i + batch], _subtree(tree, heads[h]), True) for h, hb in by_head.items() for i in range(0, len(hb), batch)]
        summary["second_pass"] = len(jobs2)
        if jobs2:
            _run(store, client, cb, kind, jobs2, valid, model, workers, effort, note + ":shortlist", summary, progress, stop)
    summary["anchor_agreement"] = anchors(store, cb)
    return summary


def _run(store, client, cb, kind, jobs, valid, model, workers, effort, note, summary, progress, stop) -> None:
    """jobs: (batch, tree to show, second_pass). Writes each batch as its reply lands."""
    client.stop = stop
    sem = _sem(resolve(model, client.base_url).base_url, max(workers, 1))
    extra = reasoning_low(model, client.base_url) if effort == "low" else None

    def one(k: int):
        if stop.is_set():
            return k, None
        b, tree, _ = jobs[k]
        with sem:
            if stop.is_set():
                return k, None
            return k, client.complete(P.assign(kind, tree, b), model=model, max_tokens=MAX_TOKENS, extra_body=extra, stage="library", note=note, system=P.SYSTEM, schema=ASSIGN_SCHEMA)

    done = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [ex.submit(one, k) for k in range(len(jobs))]
            for fut in as_completed(futs):
                k, r = fut.result()
                if r is None:
                    continue
                if r.error and r.error.startswith("denied"):
                    stop.set()
                    raise RuntimeError(r.error)
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
                        rid = parse_id(a.get("id"), {d["id"] for d in b})
                        if rid is None:
                            continue
                        fid = parse_id(a.get("feature"), valid) if a.get("feature") not in (None, "", "null") else None
                        conf = str(a.get("confidence") or "medium").lower()
                        got[rid] = (fid, conf if conf in ("high", "medium", "low") else "medium")
                    with store.lock:
                        for d in b:
                            fid, conf = got.get(d["id"], (None, "low"))     # a declaration the reply skipped is a leftover at low confidence
                            if second and fid is None:
                                continue                                    # the second pass only adds; a leftover stays as the first pass left it
                            store.con.execute("INSERT OR REPLACE INTO assignment (realization, codebook, feature, confidence, at) VALUES (?,?,?,?,?)", (d["id"], cb, fid, conf, now()))
                            if second:
                                summary["second_pass_assigned"] += 1; summary["leftover"] -= 1; summary["assigned"] += 1
                            else:
                                summary["assigned" if fid else "leftover"] += 1
                        store.con.commit()
                if progress:
                    progress(done, len(jobs), {"batch": k, "second_pass": second, "unparsed": obj is None, "assigned": summary["assigned"], "leftover": summary["leftover"], "cost": r.cost, "calls": 1})
                if stop.is_set():
                    summary["stopped"] = True
                    ex.shutdown(wait=False, cancel_futures=True)
                    client.close()
                    break
    finally:
        client.stop = None

