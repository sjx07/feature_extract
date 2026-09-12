"""Acting on the judge without rewriting anything: a flagged member (a misfit, or a member of a proposed split) goes back
to open, with a note naming the node it left. From there the shortlist pass re-decides it with the judge's reason in
front of the assigner and the node it left among the candidates: the assigner moves it, puts it back (then the flag is
standing), or leaves it open, and what fits nowhere clusters with its neighbours and reaches the naming call. Nodes, definitions and anchors are untouched, so nothing drifts; a literal flag costs a correct member one
round in the open. A node's own anchors are never reopened (a flag on one is a report on the node's definition),
and a flag raised again on the same member and node after it was acted on is standing: the member stays, the flag
is the report, and it is never reopened again. The loop ends when every flag is standing and no candidate remains.
Indistinct pairs are a report only: acting on one would merge nodes."""
from __future__ import annotations

import json

from ..store import Store, now


def reopen(store: Store, cb: int) -> dict:
    anchors = {(int(f["id"]), int(e)) for f in store.rows("SELECT id, examples FROM feature WHERE codebook=?", (cb,)) for e in json.loads(f["examples"] or "[]")}
    misfits = [(int(r["feature"]), int(r["realization"]), (r["note"] or "")[:200]) for r in store.rows("SELECT feature, realization, note FROM flag WHERE codebook=? AND verdict='misfit' AND realization IS NOT NULL AND standing=0", (cb,))
               if (int(r["feature"]), int(r["realization"])) not in anchors]                  # a node's own anchors stay: a flag on one is a report on the node
    splits = []
    for r in store.rows("SELECT feature, note FROM flag WHERE codebook=? AND verdict='split' AND standing=0", (cb,)):
        n = json.loads(r["note"] or "{}")
        for part in n.get("parts", []):
            splits += [(int(r["feature"]), int(m), f"the judge sees a distinct sub-feature here ({part.get('name', '')}): {n.get('why', '')}"[:200]) for m in part.get("members", []) if (int(r["feature"]), int(m)) not in anchors]
    n_mis = n_split = 0
    with store.lock:
        for fid, rid, why in misfits:
            n_mis += store.con.execute("UPDATE assignment SET feature=NULL, confidence='low', note=?, at=? WHERE codebook=? AND realization=? AND feature=?", (f"reopened:F{fid}|{why}", now(), cb, rid, fid)).rowcount
        for fid, rid, why in splits:
            n_split += store.con.execute("UPDATE assignment SET feature=NULL, confidence='low', note=?, at=? WHERE codebook=? AND realization=? AND feature=?", (f"split:F{fid}|{why}", now(), cb, rid, fid)).rowcount
        store.con.commit()
    return {"codebook": cb, "reopened_misfits": n_mis, "reopened_split_members": n_split, "anchors_kept": len([1 for r in store.rows("SELECT feature, realization FROM flag WHERE codebook=? AND verdict='misfit'", (cb,)) if (int(r["feature"]), int(r["realization"] or 0)) in anchors])}


def judged(store: Store, cb: int) -> dict[int, tuple[int, str]]:
    """realization -> (the node it was reopened from, the judge's reason); the assigner sees both and adjudicates."""
    out = {}
    for r in store.rows("SELECT realization, note FROM assignment WHERE codebook=? AND feature IS NULL AND (note LIKE 'reopened:F%' OR note LIKE 'split:F%')", (cb,)):
        body = r["note"].split(":F", 1)[1]
        fid, _, why = body.partition("|")
        out[int(r["realization"])] = (int(fid), why)
    return out


def settle(store: Store, cb: int, rid: int, fid: int) -> None:
    """The assigner put a reopened member back on the node the judge removed it from: the flag is standing."""
    with store.lock:
        store.con.execute("UPDATE flag SET standing=1 WHERE codebook=? AND realization=? AND feature=? AND verdict='misfit'", (cb, rid, fid))
        store.con.commit()
