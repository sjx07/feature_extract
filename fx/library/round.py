"""One round of the library loop, and the loop itself, as a single resumable sequence:

    cold start (only when the corpus has no codebook)
    assign the latest version          (skipped when it is fully assigned)
    judge it                           (skipped when it already has flags)
    revise -> the next version
    assign the new version

A corpus that grows (more prompts decomposed after a codebook exists) needs no new cold start: collapse adds the new
wordings, assign covers only those, the judge reruns, and revise sees the new leftovers.

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


class Stopped(Exception):
    pass


def run_round(store: Store, client: Client, corpus: str, kind: str, *, batch_model: str = DEFAULT_MODEL, codebook_model: str = COLDSTART_MODEL, workers: int = 128,
              effort: str = "low", rounds: int = 3, min_gain: float = MIN_GAIN, min_anchors: float = MIN_ANCHORS,
              log: Optional[Callable[[str, dict], None]] = None, progress=None, stop: Optional[threading.Event] = None) -> dict:
    """The loop as written: assign, judge, revise, assign, stop rule. Every step is idempotent (assign resumes and does
    nothing on an assigned version; judge replaces its own flags), so the loop never asks what is already done."""
    stop = stop or threading.Event()
    steps: list[dict] = []

    def step(name: str, fn) -> dict:
        r = fn()
        steps.append({"step": name, **{k: v for k, v in r.items() if k != "notes"}})
        if log:
            log(name, r)
        if r.get("stopped") or stop.is_set():
            raise Stopped()
        return r

    def coverage(version: int) -> float:
        v = [x for x in status(store, corpus, kind)["versions"] if x["version"] == version]
        return (v[0]["reading_coverage"] if v else None) or 0.0

    def do_assign(version: int) -> dict:
        return step("assign", lambda: assign(store, client, corpus, kind, model=batch_model, version=version, workers=workers, effort=effort, progress=progress, stop=stop))

    why = f"{rounds} rounds done"
    try:
        step("collapse", lambda: collapse(store, corpus, kind))
        if latest(store, corpus, kind) is None:
            step("coldstart", lambda: coldstart(store, client, corpus, kind, model=codebook_model))
        for _ in range(rounds):
            v = int(latest(store, corpus, kind)["version"])
            do_assign(v)
            step("judge", lambda: judge(store, client, corpus, kind, model=batch_model, version=v, workers=workers, effort=effort, progress=progress))
            before = coverage(v)
            r = step("revise", lambda: revise(store, client, corpus, kind, model=codebook_model, version=v))
            nv = int(r["version"])
            a = do_assign(nv)
            gain, anchors = coverage(nv) - before, a.get("anchor_agreement")
            if anchors is not None and anchors < min_anchors:
                with store.lock:
                    store.con.execute("UPDATE codebook SET notes=? WHERE id=?", (f"[anchors {anchors} < {min_anchors}: read this version first] " + (r.get("notes") or ""), a["codebook"])); store.con.commit()
                why = f"anchor agreement fell to {anchors} on v{nv}"; break
            if gain < min_gain:
                why = f"v{nv} gained {gain:+.3f} reading coverage (< {min_gain})"; break
    except Stopped:
        why = "stopped"
    return {"corpus": corpus, "kind": kind, "steps": steps, "versions": status(store, corpus, kind)["versions"], "stopped_because": why}
