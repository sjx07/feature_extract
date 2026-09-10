"""Every realization onto a feature of one codebook version, or onto none, in batches. Resumable: a realization already
assigned under the version is skipped, a batch whose reply does not parse is left for the next run. Runs from scratch
per version, so no assignment depends on the path that produced it. Ends by measuring the version's anchor agreement."""
from __future__ import annotations

import threading
from typing import Callable, Optional

from ..llm import Client
from ..llm.pool import run_many
from ..llm.registry import DEFAULT_MODEL
from ..store import Store, now
from ..util.ids import parse_id
from ..util.jsonx import extract_object
from . import prompts as P
from .codebook import ASSIGN_SCHEMA, BATCH, MAX_TOKENS, anchors, groups, latest
from .collapse import realizations


def assign(store: Store, client: Client, corpus: str, kind: str, model: str = DEFAULT_MODEL, version: Optional[int] = None, workers: int = 16, batch: int = BATCH,
           progress: Optional[Callable[[int, int, dict], None]] = None, stop: Optional[threading.Event] = None) -> dict:
    """Every realization not yet assigned under the version, in batches; a batch whose reply does not parse is left for the next run."""
    cbrow = latest(store, corpus, kind, version)
    if not cbrow:
        raise ValueError(f"no codebook for {corpus} {kind}: run coldstart first")
    cb = int(cbrow["id"])
    tree = groups(store, cb)
    valid = {f["id"] for g in tree for f in g["features"]}
    done_ids = {int(r["realization"]) for r in store.rows("SELECT realization FROM assignment WHERE codebook=?", (cb,))}
    todo = [d for d in realizations(store, corpus, kind) if d["id"] not in done_ids]
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    summary = {"codebook": cb, "version": cbrow["version"], "realizations": len(todo), "batches": len(batches), "assigned": 0, "leftover": 0, "unparsed": 0, "stopped": False}
    if not batches:
        anchors(store, cb)
        return summary
    stop = stop or threading.Event()
    client.stop = stop
    prompts_ = [P.assign(kind, tree, b) for b in batches]
    done = 0

    def on_progress(k, total):
        pass
    try:
        replies = run_many(client, prompts_, model=model, workers=workers, max_inflight=workers, stage="library", note=f"{corpus}:{kind}:assign:v{cbrow['version']}",
                           system=P.SYSTEM, max_tokens=MAX_TOKENS, schema=ASSIGN_SCHEMA, progress=on_progress)
    finally:
        client.stop = None
    for b, r in zip(batches, replies):
        done += 1
        obj = extract_object(r.text, "assignments") if r and r.text else None
        if obj is None:
            summary["unparsed"] += 1
            if progress:
                progress(done, len(batches), {"batch": done, "unparsed": True})
            continue
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
                store.con.execute("INSERT OR REPLACE INTO assignment (realization, codebook, feature, confidence, at) VALUES (?,?,?,?,?)", (d["id"], cb, fid, conf, now()))
                summary["assigned" if fid else "leftover"] += 1
            store.con.commit()
        if progress:
            progress(done, len(batches), {"batch": done, "assigned": summary["assigned"], "leftover": summary["leftover"]})
        if stop.is_set():
            summary["stopped"] = True
            break
    summary["anchor_agreement"] = anchors(store, cb)
    return summary
