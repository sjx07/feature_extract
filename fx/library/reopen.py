"""Acting on the judge without rewriting anything: a flagged member (a misfit, or a member of a proposed split) goes back
to open, with a note naming the node it left. From there it takes the path every open wording takes: the shortlist
pass re-decides it with that node excluded, and what fits nowhere clusters with its neighbours and reaches the naming
call. Nodes, definitions and anchors are untouched, so nothing drifts; a literal flag costs a correct member one
round in the open. A node's own anchors are never reopened (a flag on one is a report on the node's definition),
and the loop's last judge is not followed by a reopen, so the finished tree's flags are the report to read.
Indistinct pairs are a report only: acting on one would merge nodes."""
from __future__ import annotations

import json

from ..store import Store, now


def reopen(store: Store, cb: int) -> dict:
    anchors = {(int(f["id"]), int(e)) for f in store.rows("SELECT id, examples FROM feature WHERE codebook=?", (cb,)) for e in json.loads(f["examples"] or "[]")}
    misfits = [(int(r["feature"]), int(r["realization"])) for r in store.rows("SELECT feature, realization FROM flag WHERE codebook=? AND verdict='misfit' AND realization IS NOT NULL", (cb,))
               if (int(r["feature"]), int(r["realization"])) not in anchors]                  # a node's own anchors stay: a flag on one is a report on the node
    splits = []
    for r in store.rows("SELECT feature, note FROM flag WHERE codebook=? AND verdict='split'", (cb,)):
        for part in json.loads(r["note"] or "{}").get("parts", []):
            splits += [(int(r["feature"]), int(m)) for m in part.get("members", []) if (int(r["feature"]), int(m)) not in anchors]
    n_mis = n_split = 0
    with store.lock:
        for fid, rid in misfits:
            n_mis += store.con.execute("UPDATE assignment SET feature=NULL, confidence='low', note=?, at=? WHERE codebook=? AND realization=? AND feature=?", (f"reopened:F{fid}", now(), cb, rid, fid)).rowcount
        for fid, rid in splits:
            n_split += store.con.execute("UPDATE assignment SET feature=NULL, confidence='low', note=?, at=? WHERE codebook=? AND realization=? AND feature=?", (f"split:F{fid}", now(), cb, rid, fid)).rowcount
        store.con.commit()
    return {"codebook": cb, "reopened_misfits": n_mis, "reopened_split_members": n_split, "anchors_kept": len([1 for r in store.rows("SELECT feature, realization FROM flag WHERE codebook=? AND verdict='misfit'", (cb,)) if (int(r["feature"]), int(r["realization"] or 0)) in anchors])}


def excluded(store: Store, cb: int) -> dict[int, int]:
    """realization -> the node it was reopened from; assign must not put it back there."""
    out = {}
    for r in store.rows("SELECT realization, note FROM assignment WHERE codebook=? AND feature IS NULL AND (note LIKE 'reopened:F%' OR note LIKE 'split:F%')", (cb,)):
        out[int(r["realization"])] = int(r["note"].split(":F", 1)[1])
    return out
