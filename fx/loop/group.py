"""Step: group. Features the namer could not place under a group wait as unplaced. Once a round, one call sees all of them
beside every group and does two things: files an unplaced feature under an existing group after all, or founds a new
group for three or more unplaced features that share one purpose no group serves. A group is born from support, like a
feature; no naming call founds one from a six-item view."""
from __future__ import annotations

from typing import Callable, Optional

from ..llm import Client
from ..llm.registry import reasoning_low
from ..store import Store
from ..util.ids import parse_id, parse_ids
from ..util.jsonx import extract_object
from .calls import MAX_TOKENS
from .level import Level
from .state import tree

GROUP_SCHEMA = {"type": "object", "properties": {
    "place": {"type": "array", "items": {"type": "object", "properties": {"feature": {"type": "string"}, "group": {"type": "string"}}, "required": ["feature", "group"]}},
    "groups": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "definition": {"type": "string"}, "aspect": {"type": "string"},
                                                                          "features": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "definition", "aspect", "features"]}}},
    "required": ["place", "groups"]}


def regroup(store: Store, client: Client, lv: Level, round_: int, model: str, effort: str = "low", progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    tr = tree(store, lv)
    groups = [g for g in tr if g["id"] is not None]
    unplaced = [f for g in tr if g["id"] is None for f in g["features"]]
    summary = {"codebook": lv.codebook, "round": round_, "unplaced": len(unplaced), "placed": 0, "groups": 0, "in_new_groups": 0, "still_unplaced": len(unplaced), "unparsed": 0, "calls": 0}
    if not unplaced or not lv.prompt_group or not groups:
        return summary
    extra = reasoning_low(model, client.base_url) if effort == "low" else None
    r = client.complete(lv.prompt_group(groups, unplaced), model=model, max_tokens=MAX_TOKENS, extra_body=extra, stage="library", note=f"{lv.label}:group:r{round_}", system=lv.system, schema=GROUP_SCHEMA)
    summary["calls"] = 1
    if r.error and r.error.startswith("denied"):
        raise RuntimeError(r.error)
    obj = extract_object(r.text) if r and r.text else None
    if not isinstance(obj, dict):
        summary["unparsed"] = 1
        return summary
    fids, gids = {f["id"] for f in unplaced}, {g["id"] for g in groups}
    taken: set[int] = set()
    with store.lock:
        for p in obj.get("place") or []:
            if not isinstance(p, dict):
                continue
            fid, gid = parse_id(p.get("feature"), fids), parse_id(p.get("group"), gids)
            if fid is not None and gid is not None and fid not in taken:
                taken.add(fid)
                summary["placed"] += store.con.execute("UPDATE feature SET parent=? WHERE id=? AND parent IS NULL", (gid, fid)).rowcount
        for g in obj.get("groups") or []:
            if not isinstance(g, dict) or not str(g.get("name") or "").strip():
                continue
            members = [x for x in parse_ids(g.get("features"), fids) if x not in taken]
            if len(members) < lv.group_min_features:
                continue
            aspect = str(g.get("aspect") or "other").lower()
            gid = store.insert("feature", {"codebook": lv.codebook, "level": "group", "parent": None, "prev": None, "aspect": aspect if aspect in lv.aspects else "other",
                                           "name": str(g["name"]).strip(), "definition": str(g.get("definition") or "").strip(), "polarity": None, "examples": [], "round": round_})
            for fid in members:
                store.con.execute("UPDATE feature SET parent=? WHERE id=? AND parent IS NULL", (gid, fid))
            taken.update(members); summary["groups"] += 1; summary["in_new_groups"] += len(members)
        store.con.commit()
    summary["still_unplaced"] = len(unplaced) - len(taken)
    if progress:
        progress(1, 1, {"placed": summary["placed"], "groups": summary["groups"], "cost": r.cost, "calls": 1})
    return summary
