"""The loop, as one resumable job:

    collapse, embed                      new wordings become realizations with a vector
    cold start                           only when the corpus has no codebook
    assign                               new wordings onto the tree (a batch against the whole codebook, then the retrieval shortlist)
    judge, reopen                        the judge reports; a flagged member goes back to open, barred from the node it left
    repeat: cluster the open wordings -> name the candidates -> assign the open wordings against the tree again
            until no candidate is left, a naming round yields under three nodes, or the round limit

Nothing is rewritten: the tree only gains nodes (each stamped with its round), a wording assigned stays assigned, and
only the open wordings are ever looked at again. A wording with no neighbour in the corpus is marked specific and
skipped by naming until a new batch of prompts gives it one. Every step is one line in the job log.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

from ..llm import Client
from ..llm.registry import DEFAULT_MODEL
from ..store import Store
from .assign import assign
from .cluster import candidates
from .codebook import COLDSTART_MODEL, latest, status
from .coldstart import coldstart
from .collapse import collapse
from .embed import embed
from .judge import judge
from .name import name
from .reopen import reopen


class Stopped(Exception):
    pass


def run_round(store: Store, client: Client, corpus: str, kind: str, *, batch_model: str = DEFAULT_MODEL, codebook_model: str = COLDSTART_MODEL, workers: int = 128,
              effort: str = "low", rounds: int = 5, tau: Optional[float] = None, min_yield: int = 3, encoder=None, log: Optional[Callable[[str, dict], None]] = None, progress=None, stop: Optional[threading.Event] = None) -> dict:
    stop = stop or threading.Event()
    steps: list[dict] = []

    def step(nm: str, fn) -> dict:
        r = fn()
        steps.append({"step": nm, **{k: v for k, v in r.items() if k not in ("notes", "clusters")}})
        if log:
            log(nm, {k: v for k, v in r.items() if k != "clusters"})
        if r.get("stopped") or stop.is_set():
            raise Stopped()
        return r

    why = f"{rounds} rounds done"
    try:
        step("collapse", lambda: collapse(store, corpus, kind))
        step("embed", lambda: embed(store, corpus, kind, enc=encoder))
        if latest(store, corpus, kind) is None:
            step("coldstart", lambda: coldstart(store, client, corpus, kind, model=codebook_model))
        cb = int(latest(store, corpus, kind)["id"])
        step("assign", lambda: assign(store, client, corpus, kind, model=batch_model, workers=workers, effort=effort, progress=progress, stop=stop))
        step("judge", lambda: judge(store, client, corpus, kind, model=batch_model, workers=workers, effort=effort, progress=progress))
        step("reopen", lambda: reopen(store, cb))
        first = int(store.one("SELECT COALESCE(MAX(round), 0) r FROM feature WHERE codebook=?", (cb,))["r"]) + 1
        for rnd in range(first, first + rounds):
            c = step("cluster", lambda: candidates(store, cb, corpus, kind, tau=tau))     # tau None: measured from the anchors
            if not c["clusters"]:
                why = f"no candidate clusters left ({c['specific']} specific, {c['unclustered']} unclustered open wordings)"; break
            n = step("name", lambda: name(store, client, corpus, kind, cb, c["clusters"], rnd, model=codebook_model, workers=min(workers, 16), progress=progress))
            if n["variants"] + n["features"] < min_yield:
                why = f"round {rnd} named only {n['variants'] + n['features']} nodes from {n['clusters']} candidates (< {min_yield}): growth is done"; break
            step("assign", lambda: assign(store, client, corpus, kind, model=batch_model, workers=workers, effort=effort, only_open=True, progress=progress, stop=stop))
            step("judge", lambda: judge(store, client, corpus, kind, model=batch_model, workers=workers, effort=effort, progress=progress))
            step("reopen", lambda: reopen(store, cb))
    except Stopped:
        why = "stopped"
    return {"corpus": corpus, "kind": kind, "steps": steps, "versions": status(store, corpus, kind)["versions"], "stopped_because": why}
