"""Sorting the open wordings by whether anything else in the corpus says the same thing. No model call.

    tau = threshold(store, cb)                    measured from the features' own anchors (pairs known to share a feature)
    cands = candidates(store, cb, corpus, kind)   clusters of open wordings that neighbour each other; the rest marked specific

An open wording (assigned to no node) with no neighbour at or above tau among wordings from other prompts is
*specific* for now: its assignment note says so, and it stays out of the naming step. Open wordings that neighbour
each other form a candidate when they come from at least `min_prompts` prompts and number at least `min_members`.
Both are recomputed every round over every open wording, so a wording marked specific becomes a candidate the
moment a new batch of prompts brings it a neighbour. Similarity only proposes; nothing becomes a node until the
naming call has read the cluster and confirmed it.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

import numpy as np

from ..store import Store, now
from .codebook import latest
from .collapse import realizations
from .embed import vectors

TAU_DEFAULT, TAU_MIN, TAU_MAX = 0.78, 0.6, 0.92
MIN_MEMBERS, MIN_PROMPTS = 3, 2


def threshold(store: Store, cb: int, quantile: float = 0.25) -> tuple[float, int]:
    """The similarity at which known-same pairs (a feature's anchors) mostly count as neighbours: the quantile of
    anchor-pair cosines, clamped. Returns (tau, pairs measured); the default when fewer than 10 pairs exist."""
    sims = []
    for f in store.rows("SELECT examples FROM feature WHERE codebook=? AND level IN ('feature','variant')", (cb,)):
        ex = json.loads(f["examples"] or "[]")
        if len(ex) < 2:
            continue
        ids, m = vectors(store, ex)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sims.append(float(m[i] @ m[j]))
    if len(sims) < 10:
        return TAU_DEFAULT, len(sims)
    return float(min(TAU_MAX, max(TAU_MIN, np.quantile(sims, quantile)))), len(sims)


def open_wordings(store: Store, cb: int, corpus: str, kind: str) -> list[dict]:
    have = {int(r["realization"]): r for r in store.rows("SELECT realization, note FROM assignment WHERE codebook=? AND feature IS NULL", (cb,))}
    return [d | {"note": have[d["id"]]["note"]} for d in realizations(store, corpus, kind) if d["id"] in have]


def candidates(store: Store, cb: int, corpus: str, kind: str, tau: Optional[float] = None, min_members: int = MIN_MEMBERS, min_prompts: int = MIN_PROMPTS) -> dict:
    """Cluster the open wordings; mark the neighbourless ones specific. Returns the clusters and the counts."""
    if tau is None:
        tau, _ = threshold(store, cb)
    opened = open_wordings(store, cb, corpus, kind)
    ids, m = vectors(store, [d["id"] for d in opened])
    by_id = {d["id"]: d for d in opened}
    prompts_of: dict[int, set[str]] = defaultdict(set)
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in store.rows(f"SELECT realization, prompt FROM reading WHERE realization IN ({','.join('?' * len(chunk))})", chunk):
            prompts_of[int(r["realization"])].add(r["prompt"])
    parent = list(range(len(ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    has_neighbour = [False] * len(ids)
    if len(ids) > 1:
        sims = m @ m.T
        for i in range(len(ids)):
            row = np.where(sims[i] >= tau)[0]
            for j in row:
                if j <= i:
                    continue
                if prompts_of[ids[i]] & prompts_of[ids[j]] == prompts_of[ids[i]] | prompts_of[ids[j]]:
                    continue                                    # the same prompt saying it twice is not support
                has_neighbour[i] = has_neighbour[j] = True
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ids)):
        groups[find(i)].append(i)
    clusters = []
    for members in groups.values():
        rids = [ids[i] for i in members]
        nprompts = len(set().union(*(prompts_of[r] for r in rids)))
        if len(rids) >= min_members and nprompts >= min_prompts:
            clusters.append({"members": sorted((by_id[r] for r in rids), key=lambda d: (-d["prompts"], -d["n"])), "prompts": nprompts})
    clusters.sort(key=lambda c: (-c["prompts"], -len(c["members"])))
    specific = [ids[i] for i in range(len(ids)) if not has_neighbour[i]]
    in_cluster = {d["id"] for c in clusters for d in c["members"]}
    with store.lock:
        store.con.executemany("UPDATE assignment SET note=? WHERE codebook=? AND realization=? AND (note IS NULL OR note='specific')", [("specific", cb, r) for r in specific])
        store.con.executemany("UPDATE assignment SET note=NULL WHERE codebook=? AND realization=? AND note='specific'", [(cb, r) for r in ids if r not in specific])
        store.con.commit()
    return {"tau": round(tau, 3), "open": len(ids), "specific": len(specific), "clusters": clusters, "in_clusters": len(in_cluster), "unclustered": len(ids) - len(specific) - len(in_cluster)}
