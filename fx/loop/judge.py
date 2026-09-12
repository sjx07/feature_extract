"""Steps: judge and reopen. The judge is read-only: per node, members that do not fit and a split; per group, siblings the
members cannot tell apart; a flag raised again after it was acted on is standing. Reopen sends first-time flags' members
back to open with the reason; anchors never."""
from __future__ import annotations

import json
from ..llm import Client
from ..llm.registry import reasoning_low
from ..store import Store, now
from ..util.ids import parse_id, parse_ids
from ..util.jsonx import extract_object
from .level import Level
from typing import Callable, Optional
from .calls import _stream
from .state import members, nodes, tree

# ---- judge, reopen
def judge(store: Store, client: Client, lv: Level, model: str, workers: int = 128, effort: str = "low", progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    tr = tree(store, lv)
    jobs, meta = [], []
    for n in nodes(tr):
        ms = members(store, lv, n["id"])
        if len(ms) >= 2:
            jobs.append(lv.prompt_judge(n, ms, {m["id"]: lv.member_samples(m) for m in ms})); meta.append(("node", n, {m["id"] for m in ms}))
    if lv.prompt_siblings:
        for g in tr:
            if g["id"] is not None and len(g["features"]) >= 2:
                jobs.append(lv.prompt_siblings(g, {f["id"]: members(store, lv, f["id"], 5) for f in g["features"]})); meta.append(("group", g, {f["id"] for f in g["features"]}))
    summary = {"codebook": lv.codebook, "calls": len(jobs), "misfits": 0, "splits": 0, "indistinct": 0, "unparsed": 0, "new": 0, "standing": 0}
    previous = {(int(r["feature"]), r["realization"] and int(r["realization"]), r["other"] and int(r["other"]), r["verdict"]) for r in store.rows("SELECT feature, realization, other, verdict FROM flag WHERE codebook=?", (lv.codebook,))}
    with store.lock:
        store.con.execute("DELETE FROM flag WHERE codebook=?", (lv.codebook,)); store.con.commit()
    if not jobs:
        return summary
    member_col = "realization" if lv.kind == "realization" else "other"
    done = 0
    for k, r in _stream(client, jobs, model, workers, f"{lv.label}:judge", lv.system, extra=reasoning_low(model, client.base_url) if effort == "low" else None):
        what, node, valid = meta[k]
        done += 1
        obj = extract_object(r.text) if r and r.text else None
        if obj is None:
            summary["unparsed"] += 1; continue
        rows = []
        if what == "node":
            for m in obj.get("misfits") or []:
                uid = parse_id(m.get("id") if isinstance(m, dict) else m, valid)
                if uid is not None:
                    rows.append({"codebook": lv.codebook, "feature": node["id"], "realization": None, "other": None, "verdict": "misfit", "note": str((m.get("why") if isinstance(m, dict) else "") or "")[:300]} | {member_col: uid})
                    summary["misfits"] += 1
            sp = obj.get("split")
            if isinstance(sp, dict) and len(sp.get("parts") or []) >= 2:
                parts = [{"name": str(p.get("name") or ""), "members": parse_ids(p.get("members"), valid)} for p in sp["parts"] if isinstance(p, dict)]
                rows.append({"codebook": lv.codebook, "feature": node["id"], "realization": None, "other": None, "verdict": "split", "note": json.dumps({"why": str(sp.get("why") or "")[:300], "parts": parts})})
                summary["splits"] += 1
        else:
            for pr in obj.get("indistinct") or []:
                if isinstance(pr, dict):
                    a, b = parse_id(pr.get("a"), valid), parse_id(pr.get("b"), valid)
                    if a is not None and b is not None and a != b:
                        rows.append({"codebook": lv.codebook, "feature": a, "realization": None, "other": b, "verdict": "indistinct", "note": str(pr.get("why") or "")[:300]}); summary["indistinct"] += 1
        for row in rows:
            row["standing"] = int((row["feature"], row["realization"], row["other"], row["verdict"]) in previous)
            summary["standing" if row["standing"] else "new"] += 1
            store.insert("flag", row)
        if progress:
            progress(done, len(jobs), {"what": what, "id": node["id"], "misfits": summary["misfits"], "cost": r.cost, "calls": 1})
    return summary


def reopen(store: Store, lv: Level) -> dict:
    """First-time misfits and split members go open with the judge's reason; a node's own anchors never; standing flags stay."""
    anchor_set = {(int(f["id"]), int(e)) for f in store.rows("SELECT id, examples FROM feature WHERE codebook=?", (lv.codebook,)) for e in json.loads(f["examples"] or "[]")}
    member_col = "realization" if lv.kind == "realization" else "other"
    todo = []
    for r in store.rows(f"SELECT feature, {member_col} m, note FROM flag WHERE codebook=? AND verdict='misfit' AND standing=0 AND {member_col} IS NOT NULL", (lv.codebook,)):
        if (int(r["feature"]), int(r["m"])) not in anchor_set:
            todo.append((int(r["feature"]), int(r["m"]), (r["note"] or "")[:200], "misfit"))
    for r in store.rows("SELECT feature, note FROM flag WHERE codebook=? AND verdict='split' AND standing=0", (lv.codebook,)):
        n = json.loads(r["note"] or "{}")
        for part in n.get("parts", []):
            for m in part.get("members", []):
                if (int(r["feature"]), int(m)) not in anchor_set:
                    todo.append((int(r["feature"]), int(m), f"the judge sees a distinct sub-feature here ({part.get('name', '')}): {n.get('why', '')}"[:200], "split"))
    counts = {"misfit": 0, "split": 0}
    with store.lock:
        for nid, uid, why, what in todo:
            counts[what] += store.con.execute("UPDATE membership SET node=NULL, confidence='low', note=?, at=? WHERE kind=? AND codebook=? AND unit=? AND node=?",
                                              (f"reopened:{nid}|{why}", now(), lv.kind, lv.codebook, uid, nid)).rowcount
        store.con.commit()
    return {"codebook": lv.codebook, "reopened_misfits": counts["misfit"], "reopened_split_members": counts["split"]}
