"""Stage 3, the seed library: the per-corpus libraries aligned into global features, the loop of fx.loop.engine on the
feature level (fx.align.level). This package adds the cards (the units) and the seed codebook, and keeps the kind-only
signatures the CLI, the site and the tests use."""
from __future__ import annotations

import threading
from typing import Optional

from .. import loop as E
from ..library.codebook import COLDSTART_MODEL
from ..llm.registry import DEFAULT_MODEL
from ..store import Store
from .cards import cards, libraries
from .level import TAU, feature_level
from .seed import globals_, open_cards, regroup_by_aspect, seed_codebook, status

__all__ = ["assign", "candidates", "cards", "embed", "feature_level", "globals_", "judge", "libraries", "name", "open_cards", "regroup_by_aspect", "reopen", "run_round", "seed_codebook", "status", "threshold"]


def embed(store: Store, kind: str, enc=None, model: Optional[str] = None) -> dict:
    return E.embed(store, feature_level(store, kind), enc=enc, model=model)


def threshold(store: Store, kind: str) -> tuple[float, int]:
    """(tau, same-name pairs across corpora): the constant, and a count that says whether the libraries overlap at all."""
    from collections import defaultdict
    by_name: dict[tuple, int] = defaultdict(int)
    for c in cards(store, kind):
        by_name[(c["polarity"], c["name"].strip().lower())] += 1
    return TAU, sum(n * (n - 1) // 2 for n in by_name.values() if n >= 2)


def assign(store: Store, client, kind: str, model: str = DEFAULT_MODEL, workers: int = 64, batch: Optional[int] = None, effort: str = "low", only_open: bool = False,
           shortlist: bool = True, progress=None, stop: Optional[threading.Event] = None) -> dict:
    return E.assign(store, client, feature_level(store, kind), model, workers=workers, batch=batch, effort=effort, only_open=only_open, shortlist=shortlist, progress=progress, stop=stop)


def candidates(store: Store, kind: str, tau: Optional[float] = None, **_) -> dict:
    return E.candidates(store, feature_level(store, kind), tau=tau)


def name(store: Store, client, kind: str, clusters: list[dict], round_: int, model: str = COLDSTART_MODEL, workers: int = 16, progress=None) -> dict:
    return E.name(store, client, feature_level(store, kind), clusters, round_, model, workers=workers, progress=progress)


def judge(store: Store, client, kind: str, model: str = DEFAULT_MODEL, workers: int = 64, effort: str = "low", progress=None) -> dict:
    return E.judge(store, client, feature_level(store, kind), model, workers=workers, effort=effort, progress=progress)


def reopen(store: Store, kind: str) -> dict:
    return E.reopen(store, feature_level(store, kind))


def run_round(store: Store, client, kind: str = "guidance", *, batch_model: str = DEFAULT_MODEL, codebook_model: str = COLDSTART_MODEL, workers: int = 64, effort: str = "low",
              rounds: int = 5, tau: Optional[float] = None, min_yield: int = 2, encoder=None, log=None, progress=None, stop: Optional[threading.Event] = None) -> dict:
    r = E.run_round(store, client, feature_level(store, kind), batch_model=batch_model, codebook_model=codebook_model, workers=workers, effort=effort, rounds=rounds, tau=tau,
                    min_yield=min_yield, encoder=encoder, log=log, progress=progress, stop=stop)
    return {"kind": kind, **r, "status": status(store, kind)}
