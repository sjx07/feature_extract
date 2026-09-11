"""One round of the library loop, and the loop itself, as a single resumable sequence:

    cold start (only when the corpus has no codebook)
    assign the latest version          (skipped when it is fully assigned)
    judge it                           (skipped when it already has flags)
    revise -> the next version
    assign the new version

A corpus that grows (more prompts decomposed after a codebook exists) needs no new cold start: collapse adds the new
wordings, assign covers only those, the judge reruns because assignments changed, and revise sees the new leftovers.

then the stopping rule: the loop ends when a revision gained under `min_gain` of reading coverage, or when the new
version's anchor agreement fell under `min_anchors` (the version is kept, marked in its notes, and read first), or
after `rounds` rounds. Every step is one line in the job log (`step <name> result {...}`) so a run replays as a
sequence; every step is idempotent, so a re-run resumes at the first incomplete one at no cost for what the ledger
already holds.
"""
from __future__ import annotations

import json
import threading
from typing import Callable, Optional

from ..llm import Client
from ..llm.registry import DEFAULT_MODEL
from ..store import Store
from .assign import assign
from .codebook import COLDSTART_MODEL, latest, status
from .coldstart import coldstart
from .collapse import collapse
from .judge import judge
from .revise import revise

MIN_GAIN = 0.02        # of reading coverage, per revision
MIN_ANCHORS = 0.9


def _cov(store: Store, corpus: str, kind: str, version: int) -> Optional[float]:
    v = [x for x in status(store, corpus, kind)["versions"] if x["version"] == version]
    return v[0]["reading_coverage"] if v else None


def run_round(store: Store, client: Client, corpus: str, kind: str, *, batch_model: str = DEFAULT_MODEL, codebook_model: str = COLDSTART_MODEL, workers: int = 128,
              effort: str = "low", rounds: int = 3, min_gain: float = MIN_GAIN, min_anchors: float = MIN_ANCHORS,
              log: Optional[Callable[[str, dict], None]] = None, progress=None, stop: Optional[threading.Event] = None) -> dict:
    """Returns the loop's record: the steps run with their results, the versions' coverage and anchors, and why it stopped."""
    stop = stop or threading.Event()
    steps: list[dict] = []

    def step(name: str, fn):
        r = fn()
        steps.append({"step": name, **{k: v for k, v in r.items() if k != "notes"}})
        if log:
            log(name, r)
        return r

    step("collapse", lambda: collapse(store, corpus, kind))
    if latest(store, corpus, kind) is None:
        step("coldstart", lambda: coldstart(store, client, corpus, kind, model=codebook_model))
    why = f"{rounds} rounds done"
    for k in range(rounds):
        cb = latest(store, corpus, kind)
        v = int(cb["version"])
        st = [x for x in status(store, corpus, kind)["versions"] if x["version"] == v][0]
        wrote = False
        if st["unassigned"] or st["assigned"] + st["leftover"] == 0:
            r = step("assign", lambda: assign(store, client, corpus, kind, model=batch_model, version=v, workers=workers, effort=effort, progress=progress, stop=stop))
            if r.get("stopped") or stop.is_set():
                why = "stopped"; break
            wrote = r["assigned"] + r["leftover"] > 0
        if not st["flags"] or wrote:                    # new assignments (an incremental batch of prompts) need a fresh judgement
            step("judge", lambda: judge(store, client, corpus, kind, model=batch_model, version=v, workers=workers, effort=effort, progress=progress))
            if stop.is_set():
                why = "stopped"; break
        before = _cov(store, corpus, kind, v) or 0.0
        r = step("revise", lambda: revise(store, client, corpus, kind, model=codebook_model, version=v))
        nv = int(r["version"])
        a = step("assign", lambda: assign(store, client, corpus, kind, model=batch_model, version=nv, workers=workers, effort=effort, progress=progress, stop=stop))
        if a.get("stopped") or stop.is_set():
            why = "stopped"; break
        after = _cov(store, corpus, kind, nv) or 0.0
        anchors = a.get("anchor_agreement")
        if anchors is not None and anchors < min_anchors:
            with store.lock:
                store.con.execute("UPDATE codebook SET notes=? WHERE id=?", (f"[anchors {anchors} < {min_anchors}: read this version first] " + (r.get("notes") or ""), a["codebook"])); store.con.commit()
            why = f"anchor agreement fell to {anchors} on v{nv}"; break
        if after - before < min_gain:
            why = f"v{nv} gained {after - before:+.3f} reading coverage (< {min_gain})"; break
    return {"corpus": corpus, "kind": kind, "steps": steps, "versions": status(store, corpus, kind)["versions"], "stopped_because": why}
