"""Stage 2, the feature library of a corpus, one per kind (guidance, material): a tree that only grows. The loop is
fx.loop on the wording level (fx.library.level); this package adds what only the wording level has, collapse
and the cold start, and keeps the corpus-and-kind signatures the CLI, the site and the tests use.

    collapse(store, corpus, kind)                     readings -> realizations, no calls
    embed(store, corpus, kind)                        one vector per realization, a local model
    coldstart(store, client, corpus, kind, model)     one call over every wording -> the codebook (groups, features)
    assign / judge / reopen / candidates / name       the engine's steps on the corpus's latest codebook
    run_round(store, client, corpus, kind, ...)       collapse, embed, cold start if none, then the settle loop
"""
from __future__ import annotations

import threading
from typing import Optional

from .. import loop as E
from ..llm.registry import DEFAULT_MODEL
from ..store import Store
from .codebook import BATCH, COLDSTART_MODEL, KINDS, MIN_SUPPORT, anchors, flags, groups, latest, leftovers, members, nodes, status
from .coldstart import coldstart
from .collapse import collapse, realizations
from .level import wording_level

__all__ = ["BATCH", "COLDSTART_MODEL", "KINDS", "MIN_SUPPORT", "anchors", "assign", "candidates", "coldstart", "collapse", "embed", "flags", "groups", "judge", "latest", "regroup", "relook",
           "leftovers", "members", "name", "nodes", "preview", "realizations", "reopen", "run_round", "status", "threshold", "vectors", "wording_level"]


def _level(store: Store, corpus: str, kind: str, version: Optional[int] = None):
    cb = latest(store, corpus, kind, version)
    if not cb:
        raise ValueError(f"no codebook for {corpus} {kind}: run coldstart first")
    return wording_level(store, int(cb["id"]))


def relook(store: Store, corpus: str, kind: str, version: Optional[int] = None) -> dict:
    """Clear the 'specific' marks of a library's open wordings so the next round gives each its neighbourhood look. For
    libraries built before the loop grew by neighbourhoods, whose marks came from a cosine cutoff, not from a read."""
    lv = _level(store, corpus, kind, version)
    with store.lock:
        n = store.con.execute("UPDATE membership SET note=NULL WHERE kind='realization' AND codebook=? AND node IS NULL AND note='specific'", (lv.codebook,)).rowcount
        store.con.commit()
    return {"codebook": lv.codebook, "unmarked": n}


def embed(store: Store, corpus: str, kind: str, enc=None, model: Optional[str] = None) -> dict:
    cb = latest(store, corpus, kind)
    return E.embed(store, wording_level(store, int(cb["id"]) if cb else None, corpus, kind), enc=enc, model=model)


def vectors(store: Store, ids: list[int]):
    return E.vectors(store, "realization", ids)


def assign(store: Store, client, corpus: str, kind: str, model: str = DEFAULT_MODEL, version: Optional[int] = None, workers: int = 128, batch: int = BATCH,
           only_open: bool = False, effort: str = "low", shortlist: bool = True, progress=None, stop: Optional[threading.Event] = None) -> dict:
    return E.assign(store, client, _level(store, corpus, kind, version), model, workers=workers, batch=batch, effort=effort, only_open=only_open, shortlist=shortlist, progress=progress, stop=stop)


def regroup(store: Store, client, corpus: str, kind: str, model: str = COLDSTART_MODEL, version: Optional[int] = None, effort: str = "low", progress=None) -> dict:
    lv = _level(store, corpus, kind, version)
    rnd = int(store.one("SELECT COALESCE(MAX(round), 0) r FROM feature WHERE codebook=?", (lv.codebook,))["r"])
    return E.regroup(store, client, lv, rnd, model, effort=effort, progress=progress)


def judge(store: Store, client, corpus: str, kind: str, model: str = DEFAULT_MODEL, version: Optional[int] = None, workers: int = 128, effort: str = "low", progress=None) -> dict:
    return E.judge(store, client, _level(store, corpus, kind, version), model, workers=workers, effort=effort, progress=progress)


def reopen(store: Store, cb: int) -> dict:
    return E.reopen(store, wording_level(store, cb))


def candidates(store: Store, cb: int, corpus: str, kind: str, **_) -> dict:
    return E.candidates(store, wording_level(store, cb))


def threshold(store: Store, cb: int) -> tuple[Optional[float], int]:
    lv = wording_level(store, cb)
    return None, 0


def name(store: Store, client, corpus: str, kind: str, cb: int, clusters: list[dict], round_: int, model: str = COLDSTART_MODEL, workers: int = 16, progress=None) -> dict:
    return E.name(store, client, wording_level(store, cb), clusters, round_, model, workers=workers, progress=progress)


def run_round(store: Store, client, corpus: str, kind: str, *, batch_model: str = DEFAULT_MODEL, codebook_model: str = COLDSTART_MODEL, workers: int = 128, effort: str = "low",
              rounds: int = 5, encoder=None, log=None, progress=None, stop: Optional[threading.Event] = None) -> dict:
    def before(step):
        step("collapse", lambda: collapse(store, corpus, kind))
        if latest(store, corpus, kind) is None:
            step("coldstart", lambda: coldstart(store, client, corpus, kind, model=codebook_model))

    r = E.run_round(store, client, lambda: _level(store, corpus, kind), batch_model=batch_model, codebook_model=codebook_model, workers=workers, effort=effort, rounds=rounds,
                    encoder=encoder, before=before, log=log, progress=progress, stop=stop)
    return {"corpus": corpus, "kind": kind, **r, "versions": status(store, corpus, kind)["versions"]}


from .preview import preview  # noqa: E402
