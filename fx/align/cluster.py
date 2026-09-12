"""Open cards that neighbour cards from other corpora form candidates; an open card with no neighbour in any other corpus
is domain-specific for now (reversible when a new corpus arrives). The threshold comes from known-same pairs: features
with the same name in different corpora; with too few, a default."""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from ..store import Store, now
from .cards import cards
from .embed import vectors
from .seed import open_cards

TAU_DEFAULT, TAU_MIN, TAU_MAX = 0.86, 0.7, 0.95


def threshold(store: Store, kind: str, quantile: float = 0.25) -> tuple[float, int]:
    by_name: dict[tuple, list[int]] = defaultdict(list)
    for c in cards(store, kind):
        by_name[(c["polarity"], c["name"].strip().lower())].append(c["id"])
    sims = []
    for ids in by_name.values():
        if len(ids) < 2:
            continue
        got, m = vectors(store, ids)
        for i in range(len(got)):
            for j in range(i + 1, len(got)):
                sims.append(float(m[i] @ m[j]))
    if len(sims) < 10:
        return TAU_DEFAULT, len(sims)
    return float(min(TAU_MAX, max(TAU_MIN, np.quantile(sims, quantile)))), len(sims)


def candidates(store: Store, kind: str, tau: Optional[float] = None, min_corpora: int = 2) -> dict:
    if tau is None:
        tau, _ = threshold(store, kind)
    opened = open_cards(store, kind)
    ids, m = vectors(store, [c["id"] for c in opened])
    by_id = {c["id"]: c for c in opened}
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
                if j <= i or by_id[ids[i]]["corpus"] == by_id[ids[j]]["corpus"] or by_id[ids[i]]["polarity"] != by_id[ids[j]]["polarity"]:
                    continue                        # same corpus is not cross-corpus support; polarity is identity
                has_neighbour[i] = has_neighbour[j] = True
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ids)):
        groups[find(i)].append(i)
    clusters = []
    for mem in groups.values():
        cs = [by_id[ids[i]] for i in mem]
        if len({c["corpus"] for c in cs}) >= min_corpora:
            clusters.append({"members": sorted(cs, key=lambda c: (-c["support"], c["corpus"])), "corpora": len({c["corpus"] for c in cs})})
    clusters.sort(key=lambda c: (-c["corpora"], -len(c["members"])))
    specific = [ids[i] for i in range(len(ids)) if not has_neighbour[i]]
    with store.lock:
        for fid in ids:
            store.con.execute("INSERT OR IGNORE INTO alignment (feature, global, confidence, note, at) VALUES (?,NULL,NULL,NULL,?)", (fid, now()))
        store.con.executemany("UPDATE alignment SET note='domain-specific' WHERE feature=? AND global IS NULL AND (note IS NULL OR note='domain-specific')", [(f,) for f in specific])
        store.con.executemany("UPDATE alignment SET note=NULL WHERE feature=? AND note='domain-specific'", [(f,) for f in ids if f not in specific])
        store.con.commit()
    return {"tau": round(tau, 3), "open": len(ids), "domain_specific": len(specific), "clusters": clusters, "in_clusters": sum(len(c["members"]) for c in clusters)}
