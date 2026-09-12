"""One call per candidate cluster: the same global feature, named and defined across domains, or a rejection. An accepted
cluster becomes one global feature in the seed (under an existing or new group), and the members the model kept are
aligned to it."""
from __future__ import annotations

from typing import Callable, Optional

from ..library.codebook import COLDSTART_MODEL
from ..library.prompts import ASPECTS_GUIDANCE, ASPECTS_MATERIAL
from ..llm import Client
from ..llm.pool import run_many
from ..store import Store, now
from ..util.ids import parse_id, parse_ids
from ..util.jsonx import extract_object
from . import prompts as P
from .seed import globals_, seed_codebook

MAX_TOKENS = 32768
SCHEMA = {"type": "object", "properties": {"decision": {"type": "string", "enum": ["same", "reject"]}, "why": {"type": "string"},
          "group": {"anyOf": [{"type": "string"}, {"type": "null"}, {"type": "object", "properties": {"name": {"type": "string"}, "definition": {"type": "string"}, "aspect": {"type": "string"}}, "required": ["name", "definition", "aspect"], "additionalProperties": False}]},
          "name": {"type": "string"}, "definition": {"type": "string"}, "polarity": {"type": "string", "enum": ["require", "forbid"]}, "members": {"type": "array", "items": {"type": "string"}}},
          "required": ["decision", "why", "group", "name", "definition", "polarity", "members"], "additionalProperties": False}


def name(store: Store, client: Client, kind: str, clusters: list[dict], round_: int, model: str = COLDSTART_MODEL, workers: int = 16,
         progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    cb = seed_codebook(store, kind)
    tree = globals_(store, kind)
    group_ids = {g["id"]: g for g in tree}
    aspects = ASPECTS_GUIDANCE if kind == "guidance" else ASPECTS_MATERIAL
    summary = {"codebook": cb, "round": round_, "clusters": len(clusters), "globals": 0, "groups": 0, "rejected": 0, "unparsed": 0, "aligned": 0}
    if not clusters:
        return summary
    replies = run_many(client, [P.name(kind, tree, c["members"]) for c in clusters], model=model, workers=workers, max_inflight=workers, stage="align",
                       note=f"align:{kind}:name:r{round_}", system=P.SYSTEM, max_tokens=MAX_TOKENS, schema=SCHEMA)
    for k, (c, r) in enumerate(zip(clusters, replies)):
        obj = extract_object(r.text) if r and r.text else None
        member_ids = {m["id"] for m in c["members"]}
        by_id = {m["id"]: m for m in c["members"]}
        if not isinstance(obj, dict) or obj.get("decision") not in ("same", "reject"):
            summary["unparsed"] += 1; continue
        kept = parse_ids(obj.get("members"), member_ids)
        if obj["decision"] == "reject" or len({by_id[f]["corpus"] for f in kept}) < 2 or not str(obj.get("name") or "").strip():
            summary["rejected"] += 1
            if progress:
                progress(k + 1, len(clusters), {"cluster": k, "decision": "reject"})
            continue
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
            summary["rejected"] += 1; continue
        pol = str(obj.get("polarity") or "require").lower()
        sid = store.insert("feature", {"codebook": cb, "level": "feature", "parent": gid, "prev": None, "aspect": None, "name": str(obj["name"]).strip(), "definition": str(obj.get("definition") or "").strip(),
                                       "polarity": pol if pol in ("require", "forbid") else "require", "examples": kept[:3], "round": round_})
        with store.lock:
            for fid in kept:
                store.con.execute("INSERT INTO alignment (feature, global, confidence, note, at) VALUES (?,?,'high','named',?) ON CONFLICT(feature) DO UPDATE SET global=excluded.global, confidence='high', note='named', at=excluded.at WHERE alignment.global IS NULL", (fid, sid, now()))
            store.con.commit()
        summary["globals"] += 1; summary["aligned"] += len(kept)
        if progress:
            progress(k + 1, len(clusters), {"cluster": k, "decision": "same", "name": obj.get("name")})
    return summary
