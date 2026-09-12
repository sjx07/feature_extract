"""The align loop, one resumable job:

    embed the cards           every per-corpus feature gets a vector
    assign                    open cards onto the globals nearest to them, or none
    judge                     reports
    repeat: reopen the judge's first-time flags -> cluster the open cards across corpora -> name the candidates
            -> assign the open cards again -> judge
            until settled: no new flag and no candidate left (or the round limit)

Nothing in a corpus library changes; the seed only gains globals and alignments. A card with no neighbour in another
corpus is domain-specific for now and is re-examined whenever a corpus is added.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

from ..library.codebook import COLDSTART_MODEL
from ..llm import Client
from ..llm.registry import DEFAULT_MODEL
from ..store import Store
from .assign import assign
from .cluster import candidates
from .embed import embed
from .judge import judge, reopen
from .name import name
from .seed import seed_codebook, status


class Stopped(Exception):
    pass


def run_round(store: Store, client: Client, kind: str = "guidance", *, batch_model: str = DEFAULT_MODEL, codebook_model: str = COLDSTART_MODEL, workers: int = 64,
              effort: str = "low", rounds: int = 5, tau: Optional[float] = None, min_yield: int = 2, encoder=None,
              log: Optional[Callable[[str, dict], None]] = None, progress=None, stop: Optional[threading.Event] = None) -> dict:
    stop = stop or threading.Event()
    steps: list[dict] = []

    def step(nm: str, fn) -> dict:
        r = fn()
        steps.append({"step": nm, **{k: v for k, v in r.items() if k != "clusters"}})
        if log:
            log(nm, {k: v for k, v in r.items() if k != "clusters"})
        if stop.is_set():
            raise Stopped()
        return r

    why = f"{rounds} rounds done"
    try:
        cb = seed_codebook(store, kind)
        step("embed", lambda: embed(store, kind, enc=encoder))
        step("assign", lambda: assign(store, client, kind, model=batch_model, workers=workers, effort=effort, progress=progress))
        step("judge", lambda: judge(store, client, kind, model=batch_model, workers=workers, effort=effort, progress=progress))
        first = int(store.one("SELECT COALESCE(MAX(round), 0) r FROM feature WHERE codebook=?", (cb,))["r"]) + 1
        for rnd in range(first, first + rounds):
            r = step("reopen", lambda: reopen(store, kind))
            c = step("cluster", lambda: candidates(store, kind, tau=tau))
            if not c["clusters"] and r["reopened"] == 0:
                why = f"settled: every flag is standing and no candidate cluster is left ({c['domain_specific']} domain-specific cards)"; break
            n = step("name", lambda: name(store, client, kind, c["clusters"], rnd, model=codebook_model, workers=min(workers, 16), progress=progress)) if c["clusters"] else {"globals": 0, "clusters": 0}
            step("assign", lambda: assign(store, client, kind, model=batch_model, workers=workers, effort=effort, progress=progress))
            j = step("judge", lambda: judge(store, client, kind, model=batch_model, workers=workers, effort=effort, progress=progress))
            if n["globals"] < min_yield and j["new"] == 0:
                why = f"settled: round {rnd} named {n['globals']} globals and the judge raised nothing new ({j['standing']} standing flags)"; break
    except Stopped:
        why = "stopped"
    return {"kind": kind, "steps": steps, "status": status(store, kind), "stopped_because": why}
