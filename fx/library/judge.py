"""Read-only coherence of one version: per feature, members that do not fit and a split when the members fall apart;
per group, sibling features the members cannot tell apart. Writes flags and nothing else; the next revision sees them."""
from __future__ import annotations

import json
from typing import Callable, Optional

from ..llm import Client
from ..llm.pool import run_many
from ..llm.registry import DEFAULT_MODEL, reasoning_low
from ..store import Store
from ..util.ids import parse_id, parse_ids
from ..util.jsonx import extract_object
from . import prompts as P
from .codebook import MAX_TOKENS, groups, latest, members, nodes


def judge(store: Store, client: Client, corpus: str, kind: str, model: str = DEFAULT_MODEL, version: Optional[int] = None, workers: int = 128, effort: str = "low",
          progress: Optional[Callable[[int, int, dict], None]] = None) -> dict:
    """Read-only coherence on one version: misfit members and splits per feature, indistinct pairs per group. Replaces the version's flags."""
    cbrow = latest(store, corpus, kind, version)
    if not cbrow:
        raise ValueError("no codebook")
    cb = int(cbrow["id"])
    tree = groups(store, cb)
    feats = nodes(tree)                                   # features and variants alike
    jobs, meta = [], []
    for f in feats:
        ms = members(store, cb, f["id"])
        if len(ms) >= 2:
            jobs.append(P.judge_members(f, ms)); meta.append(("feature", f, {m["id"] for m in ms}))
    for g in tree:
        if len(g["features"]) >= 2:
            samples = {f["id"]: members(store, cb, f["id"], 5) for f in g["features"]}
            jobs.append(P.judge_siblings(g, samples)); meta.append(("group", g, {f["id"] for f in g["features"]}))
    replies = run_many(client, jobs, model=model, workers=workers, max_inflight=workers, stage="library", note=f"{corpus}:{kind}:judge:v{cbrow['version']}", system=P.SYSTEM, max_tokens=MAX_TOKENS, extra_body=reasoning_low(model, client.base_url) if effort == "low" else None) if jobs else []
    summary = {"codebook": cb, "version": cbrow["version"], "calls": len(jobs), "misfits": 0, "splits": 0, "indistinct": 0, "unparsed": 0, "new": 0, "standing": 0}
    # what the previous judge said: a flag raised again on the same member and node (or the same pair) is standing
    previous = {(int(r["feature"]), r["realization"] and int(r["realization"]), r["other"] and int(r["other"]), r["verdict"]) for r in store.rows("SELECT feature, realization, other, verdict FROM flag WHERE codebook=?", (cb,))}
    with store.lock:
        store.con.execute("DELETE FROM flag WHERE codebook=?", (cb,)); store.con.commit()
    for (what, node, valid), r in zip(meta, replies):
        obj = extract_object(r.text) if r and r.text else None
        if obj is None:
            summary["unparsed"] += 1
            continue
        rows = []
        if what == "feature":
            for m in obj.get("misfits") or []:
                rid = parse_id(m.get("id") if isinstance(m, dict) else m, valid)
                if rid is not None:
                    rows.append({"codebook": cb, "feature": node["id"], "realization": rid, "other": None, "verdict": "misfit", "note": str((m.get("why") if isinstance(m, dict) else "") or "")[:300]})
            sp = obj.get("split")
            if isinstance(sp, dict) and len(sp.get("parts") or []) >= 2:
                parts = [{"name": str(p.get("name") or ""), "members": parse_ids(p.get("members"), valid)} for p in sp["parts"] if isinstance(p, dict)]
                rows.append({"codebook": cb, "feature": node["id"], "realization": None, "other": None, "verdict": "split", "note": json.dumps({"why": str(sp.get("why") or "")[:300], "parts": parts})})
                summary["splits"] += 1
            summary["misfits"] += sum(1 for x in rows if x["verdict"] == "misfit")
        else:
            for pr in obj.get("indistinct") or []:
                if not isinstance(pr, dict):
                    continue
                a, b = parse_id(pr.get("a"), valid), parse_id(pr.get("b"), valid)
                if a is not None and b is not None and a != b:
                    rows.append({"codebook": cb, "feature": a, "realization": None, "other": b, "verdict": "indistinct", "note": str(pr.get("why") or "")[:300]})
                    summary["indistinct"] += 1
        for row in rows:
            key = (row["feature"], row["realization"], row["other"], row["verdict"])
            row["standing"] = int(key in previous)
            summary["standing" if row["standing"] else "new"] += 1
            store.insert("flag", row)
        if progress:
            progress(summary["calls"], len(jobs), {"what": what, "id": node["id"]})
    return summary
