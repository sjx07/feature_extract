"""Step: join. The naming calls of a round ran in parallel, each over one neighbourhood, and could not see each other; the
join is one call that sees every proposal beside the tree and decides what the round adds: a proposal is new, or the same
instruction as an existing feature (its members go there), or a duplicate of another proposal (the two become one node).
It also places what has no group and founds a group where three unplaced features share a purpose. Then the round writes,
once. Nothing older than the round is touched: definitions never change, and no node is retired after it was born."""
from __future__ import annotations

from typing import Callable, Optional

from ..llm import Client
from ..llm.registry import reasoning_low
from ..store import Store, now
from ..util.ids import parse_id, parse_ids
from ..util.jsonx import extract_object
from .calls import JOIN_SCHEMA, MAX_TOKENS
from .level import Level
from .state import nodes, tree


def join(store: Store, client: Client, lv: Level, proposals: list[dict], round_: int, model: str, effort: str = "low",
         progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    tr = tree(store, lv)
    features = {n["id"]: n for n in nodes(tr) if n["level"] == "feature"}
    groups = {g["id"]: g for g in tr if g["id"] is not None}
    unplaced = [f for g in tr if g["id"] is None for f in g["features"]]
    summary = {"codebook": lv.codebook, "round": round_, "proposals": len(proposals), "calls": 0, "unparsed": 0, "variants": 0, "features": 0,
               "into_existing": 0, "duplicates": 0, "assigned": 0, "unplaced": 0, "placed": 0, "groups": 0, "still_unplaced": 0}
    verdict = {k: ("new", None) for k in range(len(proposals))}
    place: dict = {}
    founded: list = []
    if proposals or (unplaced and not lv.groups_fixed):
        if lv.prompt_join:
            extra = reasoning_low(model, client.base_url) if effort == "low" else None
            r = client.complete(lv.prompt_join(tr, proposals, unplaced), model=model, max_tokens=MAX_TOKENS, extra_body=extra, stage="library", note=f"{lv.label}:join:r{round_}", system=lv.system, schema=JOIN_SCHEMA)
            summary["calls"] = 1
            if r.error and r.error.startswith("denied"):
                raise RuntimeError(r.error)
            obj = extract_object(r.text) if r and r.text else None
            if not isinstance(obj, dict):
                summary["unparsed"] = 1; obj = {}
            pids = set(range(1, len(proposals) + 1))
            for v in obj.get("proposals") or []:
                if not isinstance(v, dict):
                    continue
                k = parse_id(v.get("id"), pids)
                if k is None:
                    continue
                if v.get("verdict") == "existing":
                    fid = parse_id(v.get("feature"), set(features))
                    if fid is not None and features[fid]["polarity"] == proposals[k - 1]["polarity"]:
                        verdict[k - 1] = ("existing", fid)
                elif v.get("verdict") == "duplicate":
                    of = parse_id(v.get("of"), pids)
                    if of is not None and of != k and proposals[of - 1]["polarity"] == proposals[k - 1]["polarity"]:
                        verdict[k - 1] = ("duplicate", of - 1)
            for p in obj.get("place") or []:
                if isinstance(p, dict):
                    place[str(p.get("id") or "")] = parse_id(p.get("group"), set(groups))
            founded = [g for g in obj.get("groups") or [] if isinstance(g, dict) and str(g.get("name") or "").strip()]
    # duplicates chain to a surviving proposal that is itself new
    def survivor(k: int) -> int:
        seen = set()
        while verdict[k][0] == "duplicate" and k not in seen:
            seen.add(k); k = verdict[k][1]
        return k
    for k in range(len(proposals)):
        if verdict[k][0] == "duplicate" and verdict[survivor(k)][0] != "new":
            verdict[k] = ("new", None)
    node_of: dict[int, int] = {}                                  # proposal index -> node id written or joined
    placed_units: set[int] = set()
    with store.lock:
        for k, p in enumerate(proposals):
            kind, target = verdict[k]
            if kind == "existing":
                node_of[k] = target; summary["into_existing"] += 1
            elif kind == "duplicate":
                continue
            else:
                row = {"codebook": lv.codebook, "prev": None, "aspect": None, "name": p["name"], "definition": p["definition"], "polarity": p["polarity"],
                       "examples": p["examples"], "round": round_}
                if p["parent"] is not None:
                    row |= {"level": "variant", "parent": p["parent"]}; summary["variants"] += 1
                else:
                    gid = p["group"]
                    if gid is None and lv.groups_fixed:              # the seed: the group of the aspect, else 'other'
                        gid = next((i for i, g in groups.items() if g.get("aspect") == p["aspect"]), None) or next((i for i, g in groups.items() if g.get("aspect") == "other"), None)
                    row |= {"level": "feature", "parent": gid, "aspect": p["aspect"] if gid is None else None}; summary["features"] += 1
                    if gid is None:
                        summary["unplaced"] += 1
                keys = list(row)
                cur = store.con.execute(f"INSERT INTO feature ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})", [__import__('json').dumps(row[x]) if isinstance(row[x], (list, dict)) else row[x] for x in keys])
                node_of[k] = int(cur.lastrowid)
        for k, p in enumerate(proposals):
            nid = node_of.get(k if verdict[k][0] != "duplicate" else survivor(k))
            if nid is None:
                continue
            if verdict[k][0] == "duplicate":
                summary["duplicates"] += 1
            for u in p["members"]:
                store.con.execute("INSERT INTO membership (kind, unit, codebook, node, confidence, note, at) VALUES (?,?,?,?,'high','named',?) ON CONFLICT(kind, unit, codebook) DO UPDATE SET node=excluded.node, confidence='high', note='named', at=excluded.at WHERE membership.node IS NULL",
                                  (lv.kind, u["id"], lv.codebook, nid, now()))
                placed_units.add(u["id"]); summary["assigned"] += 1
        # groups: place, then found
        pending = {f"F{f['id']}": f["id"] for f in unplaced} | {f"P{k + 1}": node_of[k] for k in range(len(proposals)) if verdict[k][0] == "new" and proposals[k]["parent"] is None and proposals[k]["group"] is None and k in node_of}
        taken: set[int] = set()
        for key, gid in place.items():
            fid = pending.get(key)
            if fid is not None and gid is not None and fid not in taken:
                taken.add(fid); summary["placed"] += store.con.execute("UPDATE feature SET parent=?, aspect=NULL WHERE id=? AND parent IS NULL", (gid, fid)).rowcount
        for g in founded:
            ids = [pending[x] for x in (g.get("ids") or []) if isinstance(x, str) and x in pending and pending[x] not in taken]
            if len(ids) < lv.group_min_features:
                continue
            aspect = str(g.get("aspect") or "other").lower()
            cur = store.con.execute("INSERT INTO feature (codebook, level, parent, prev, aspect, name, definition, polarity, examples, round) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                    (lv.codebook, "group", None, None, aspect if aspect in lv.aspects else "other", str(g["name"]).strip(), str(g.get("definition") or "").strip(), None, "[]", round_))
            for fid in ids:
                store.con.execute("UPDATE feature SET parent=?, aspect=NULL WHERE id=? AND parent IS NULL", (cur.lastrowid, fid))
            taken.update(ids); summary["groups"] += 1
        summary["still_unplaced"] = len(pending) - len(taken) if not lv.groups_fixed else 0
        # a neighbourhood whose seed was not placed has had its look
        for p in proposals:
            if p.get("seed") is not None and p["seed"] not in placed_units:
                store.con.execute("UPDATE membership SET note='specific' WHERE kind=? AND codebook=? AND unit=? AND node IS NULL AND (note IS NULL OR note='specific')", (lv.kind, lv.codebook, p["seed"]))
        store.con.commit()
    if progress and summary["calls"]:
        progress(1, 1, {"join": True, "features": summary["features"], "variants": summary["variants"], "into_existing": summary["into_existing"], "duplicates": summary["duplicates"], "groups": summary["groups"], "cost": r.cost, "calls": 1})
    return summary
