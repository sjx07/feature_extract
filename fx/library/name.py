"""Naming the candidate clusters: one call per cluster, in parallel. The model reads the members and places the cluster
as a variant under a feature, as a new feature under an existing or new group, or rejects it. An accepted cluster
becomes one node, added this round; the members the model kept are assigned to it (note 'named'). Nothing existing
is touched. Rejected members stay open, and the next round's clustering sees them again with whatever new wordings
arrived."""
from __future__ import annotations

from typing import Callable, Optional

from ..corpus import corpus_domain
from ..llm import Client
from ..llm.pool import run_many
from ..store import Store, now
from ..util.ids import parse_id, parse_ids
from ..util.jsonx import extract_object
from . import prompts as P
from .codebook import COLDSTART_MODEL, MAX_TOKENS, NAME_SCHEMA, groups, nodes


def name(store: Store, client: Client, corpus: str, kind: str, cb: int, clusters: list[dict], round_: int, model: str = COLDSTART_MODEL, workers: int = 16,
         progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    cid = int(store.one("SELECT corpus FROM codebook WHERE id=?", (cb,))["corpus"])
    domain = corpus_domain(store, cid, corpus)
    tree = groups(store, cb)
    features = {n["id"]: n for n in nodes(tree) if n["level"] == "feature"}
    group_ids = {g["id"]: g for g in tree}
    aspects = P.ASPECTS_GUIDANCE if kind == "guidance" else P.ASPECTS_MATERIAL
    summary = {"codebook": cb, "round": round_, "clusters": len(clusters), "variants": 0, "features": 0, "groups": 0, "rejected": 0, "unparsed": 0, "assigned": 0}
    if not clusters:
        return summary
    replies = run_many(client, [P.name(kind, domain, tree, c["members"]) for c in clusters], model=model, workers=workers, max_inflight=workers, stage="library",
                       note=f"{corpus}:{kind}:name:r{round_}", system=P.SYSTEM, max_tokens=MAX_TOKENS, schema=NAME_SCHEMA)
    for k, (c, r) in enumerate(zip(clusters, replies)):
        obj = extract_object(r.text) if r and r.text else None
        member_ids = {d["id"] for d in c["members"]}
        if not isinstance(obj, dict) or obj.get("decision") not in ("variant", "feature", "reject"):
            summary["unparsed"] += 1
            continue
        if obj["decision"] == "reject":
            summary["rejected"] += 1
        else:
            kept = parse_ids(obj.get("members"), member_ids)
            if len(kept) < 2 or not str(obj.get("name") or "").strip():
                summary["rejected"] += 1
            else:
                pol = str(obj.get("polarity") or "require").lower()
                row = {"codebook": cb, "prev": None, "aspect": None, "name": str(obj["name"]).strip(), "definition": str(obj.get("definition") or "").strip(),
                       "polarity": pol if pol in ("require", "forbid") else "require", "examples": parse_ids(obj.get("examples"), member_ids) or kept[:3], "round": round_}
                parent = None
                if obj["decision"] == "variant":
                    parent = parse_id(obj.get("parent"), set(features))
                    if parent is not None:
                        row |= {"level": "variant", "parent": parent}
                if parent is None:                                  # a feature, or a variant whose parent was not a feature id
                    g = obj.get("group")
                    gid = parse_id(g, set(group_ids)) if isinstance(g, str) else None
                    if gid is None and isinstance(g, dict) and str(g.get("name") or "").strip():
                        key = str(g["name"]).strip().lower()
                        gid = next((i for i, x in group_ids.items() if x["name"].strip().lower() == key), None)
                        if gid is None:
                            aspect = str(g.get("aspect") or "other").lower()
                            gid = store.insert("feature", {"codebook": cb, "level": "group", "parent": None, "prev": None, "aspect": aspect if aspect in aspects else "other",
                                                           "name": str(g["name"]).strip(), "definition": str(g.get("definition") or "").strip(), "polarity": None, "examples": [], "round": round_})
                            group_ids[gid] = {"id": gid, "name": str(g["name"]).strip()}; summary["groups"] += 1
                    if gid is None:
                        summary["rejected"] += 1
                        if progress:
                            progress(k + 1, len(clusters), {"cluster": k, "decision": "reject (no group)"})
                        continue
                    row |= {"level": "feature", "parent": gid}
                nid = store.insert("feature", row)
                summary["variants" if row["level"] == "variant" else "features"] += 1
                with store.lock:
                    store.con.executemany("UPDATE assignment SET feature=?, confidence='high', note='named', at=? WHERE codebook=? AND realization=? AND feature IS NULL", [(nid, now(), cb, rid) for rid in kept])
                    store.con.commit()
                summary["assigned"] += len(kept)
        if progress:
            progress(k + 1, len(clusters), {"cluster": k, "decision": obj["decision"], "name": obj.get("name")})
    return summary
