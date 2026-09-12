"""Steps: candidates and name. Every open unit not yet looked at gets its neighbourhood, itself and its nearest open units
from other groups (no threshold); one naming call per neighbourhood: a variant under a feature, a new feature under a
group, or a rejection, after which the seed is specific."""
from __future__ import annotations

import numpy as np
from ..llm import Client
from ..llm.registry import reasoning_low
from ..store import Store, now
from ..util.ids import parse_id, parse_ids
from ..util.jsonx import extract_object
from .level import Level
from typing import Callable, Optional
from .calls import NAME_SCHEMA, _stream
from .state import node_vectors, members, nodes, open_units, tree, vectors

# ---- candidates, name
def candidates(store: Store, lv: Level) -> dict:
    """The open units as neighbourhoods for the namer; see _neighbourhoods. No threshold anywhere in the loop: retrieval
    orders, the model decides."""
    opened = open_units(store, lv)
    ids, m = vectors(store, lv.kind, [u["id"] for u in opened])
    by_id = {u["id"]: u for u in opened}
    with store.lock:                              # every open unit has a membership row, so the marks below have somewhere to go
        for u in opened:
            if u["note"] is None:
                store.con.execute("INSERT OR IGNORE INTO membership (kind, unit, codebook, node, confidence, note, at) VALUES (?,?,?,NULL,NULL,NULL,?)", (lv.kind, u["id"], lv.codebook, now()))
        store.con.commit()
    return _neighbourhoods(store, lv, ids, m, by_id)


def _neighbourhoods(store: Store, lv: Level, ids: list[int], m: np.ndarray, by_id: dict) -> dict:
    """Candidates without a threshold. Every open unit not yet marked specific is a seed; its cluster is the seed and its k
    nearest open units from other groups with the same polarity (specific ones included, so the mark is reversible), each unit
    in one cluster a round. The namer decides what the neighbourhood holds; a rejected seed becomes specific, having had its
    look. A seed whose partners are all taken this round waits for the next. Ends when no seed is left."""
    n = len(ids)
    sims = m @ m.T if n > 1 else np.zeros((max(n, 1), max(n, 1)))

    def partner(i: int, j: int) -> bool:
        ui, uj = by_id[ids[i]], by_id[ids[j]]
        return i != j and ui["polarity"] == uj["polarity"] and ui["groups"] & uj["groups"] != ui["groups"] | uj["groups"]
    ranked = {i: [int(j) for j in np.argsort(-sims[i]) if partner(i, int(j))] for i in range(n)}
    seeds = [i for i in range(n) if by_id[ids[i]]["note"] != "specific"]
    seeds.sort(key=lambda i: -(sims[i][ranked[i][0]] if ranked[i] else -2.0))
    taken: set[int] = set(); clusters = []; waiting = 0; alone = []
    for i in seeds:
        if i in taken:
            continue
        if not ranked[i]:
            alone.append(ids[i]); continue                    # no unit of another group to compare with at all
        free = [j for j in ranked[i] if j not in taken][:lv.neighbourhood]
        if not free:
            waiting += 1; continue
        taken.add(i); taken.update(free)
        us = [by_id[ids[x]] for x in [i] + free]
        clusters.append({"members": us, "groups": len(set().union(*(u["groups"] for u in us))), "seed": ids[i]})
    with store.lock:
        store.con.executemany("UPDATE membership SET note='specific' WHERE kind=? AND codebook=? AND unit=? AND node IS NULL AND (note IS NULL OR note='specific')", [(lv.kind, lv.codebook, u) for u in alone])
        store.con.commit()
    marked = {u["id"] for u in by_id.values() if u["note"] == "specific"} | set(alone)
    in_cluster = {u["id"] for c in clusters for u in c["members"]}          # specific units sit in clusters too, as neighbours
    return {"open": n, "specific": len(marked), "seeds": len(seeds), "waiting": waiting, "clusters": clusters, "in_clusters": len(in_cluster), "unclustered": n - len(marked | in_cluster)}


def _nearest_group(store: Store, lv: Level, tr: list[dict], kept: list[int], aspect: Optional[str]) -> Optional[int]:
    """The group whose features' vectors lie nearest the mean of the kept units' vectors; restricted to `aspect` when given and
    some group carries it. None only when the tree has no group at all."""
    cands = [g for g in tr if aspect is None or g.get("aspect") == aspect] or list(tr)
    if not cands:
        return None
    ids, m = vectors(store, lv.kind, kept)
    nid, nm = node_vectors(store, lv, tr)
    if not ids or not nid:
        return cands[0]["id"]
    q = m.mean(axis=0); q = q / max(float(np.linalg.norm(q)), 1e-9)
    of = dict(zip(nid, nm))
    best, score = cands[0]["id"], -2.0
    for g in cands:
        vs = [of[f["id"]] for f in g["features"] if f["id"] in of]
        if vs:
            v = np.mean(vs, axis=0); s = float(v @ q) / max(float(np.linalg.norm(v)), 1e-9)
            if s > score:
                best, score = g["id"], s
    return best


def _seed_looked(store: Store, lv: Level, c: dict, kept: list[int]) -> None:
    """A neighbourhood's seed that the namer did not place has had its look: specific, unless it carries a judge's reason still
    waiting for the assigner."""
    if c.get("seed") is not None and c["seed"] not in kept:
        with store.lock:
            store.con.execute("UPDATE membership SET note='specific' WHERE kind=? AND codebook=? AND unit=? AND node IS NULL AND (note IS NULL OR note='specific')", (lv.kind, lv.codebook, c["seed"]))
            store.con.commit()


def name(store: Store, client: Client, lv: Level, clusters: list[dict], round_: int, model: str, workers: int = 16, effort: str = "low",
         progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    tr = tree(store, lv)
    features = {n["id"]: n for n in nodes(tr) if n["level"] == "feature"}
    group_ids = {g["id"]: g for g in tr}
    summary = {"codebook": lv.codebook, "round": round_, "clusters": len(clusters), "variants": 0, "features": 0, "placed_by_vector": 0, "rejected": 0, "unparsed": 0, "assigned": 0}
    if not clusters:
        return summary
    prompts = [lv.prompt_name(tr, c["members"]) for c in clusters]
    done = 0
    extra = reasoning_low(model, client.base_url) if effort == "low" else None       # the same knob as assign and judge, whatever the model
    for k, r in _stream(client, prompts, model, workers, f"{lv.label}:name:r{round_}", lv.system, schema=NAME_SCHEMA, extra=extra):
        c = clusters[k]
        done += 1
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
            _seed_looked(store, lv, c, [])
            if progress:
                progress(done, len(clusters), {"cluster": k, "decision": "reject", "cost": r.cost, "calls": 1})
            continue
        pol = str(obj.get("polarity") or "require").lower()
        row = {"codebook": lv.codebook, "prev": None, "aspect": None, "name": str(obj["name"]).strip(), "definition": str(obj.get("definition") or "").strip(),
               "polarity": pol if pol in ("require", "forbid") else "require", "examples": parse_ids(obj.get("examples"), member_ids) or kept[:3], "round": round_}
        parent = parse_id(obj.get("parent"), set(features)) if decision == "variant" and lv.allow_variant else None
        if parent is not None:
            row |= {"level": "variant", "parent": parent}
        else:
            # a naming call never creates a group: one neighbourhood is no view of the whole. It picks a group by id; if it
            # names an aspect or nothing usable, the feature goes to the nearest existing group by vectors (within that aspect
            # when one was named), so the groups stay the ones the cold start (or the seed's aspects) laid down
            g = obj.get("group")
            gid = parse_id(g, set(group_ids)) if isinstance(g, str) else None
            if gid is None:
                aspect = str((g.get("aspect") if isinstance(g, dict) else g) or "").lower()
                gid = _nearest_group(store, lv, tr, kept, aspect if aspect in lv.aspects else None)
                summary["placed_by_vector"] += 1
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
        _seed_looked(store, lv, c, kept)
        if progress:
            progress(done, len(clusters), {"cluster": k, "decision": row["level"], "name": row["name"], "cost": r.cost, "calls": 1})
    return summary
