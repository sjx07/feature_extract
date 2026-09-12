"""Stage 3, the seed library: the per-corpus libraries aligned into global features, the loop of fx.loop on the
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
from .level import feature_level
from .seed import globals_, open_cards, regroup_by_aspect, reset, seed_codebook, status

__all__ = ["assign", "candidates", "cards", "embed", "feature_level", "globals_", "judge", "libraries", "name", "open_cards", "regroup_by_aspect", "reopen", "reset", "run_round", "seed_codebook", "status"]


def embed(store: Store, kind: str, enc=None, model: Optional[str] = None) -> dict:
    return E.embed(store, feature_level(store, kind), enc=enc, model=model)


def assign(store: Store, client, kind: str, model: str = DEFAULT_MODEL, workers: int = 64, batch: Optional[int] = None, effort: str = "low", only_open: bool = False,
           shortlist: bool = True, progress=None, stop: Optional[threading.Event] = None) -> dict:
    return E.assign(store, client, feature_level(store, kind), model, workers=workers, batch=batch, effort=effort, only_open=only_open, shortlist=shortlist, progress=progress, stop=stop)


def candidates(store: Store, kind: str, **_) -> dict:
    return E.candidates(store, feature_level(store, kind))


def name(store: Store, client, kind: str, clusters: list[dict], round_: int, model: str = COLDSTART_MODEL, workers: int = 16, progress=None) -> dict:
    lv = feature_level(store, kind)
    n = E.name(store, client, lv, clusters, round_, model, workers=workers, progress=progress)
    j = E.join(store, client, lv, n["proposals"], round_, model, progress=progress)
    return {k: v for k, v in n.items() if k != "proposals"} | {"join": j}


def judge(store: Store, client, kind: str, model: str = DEFAULT_MODEL, workers: int = 64, effort: str = "low", progress=None) -> dict:
    return E.judge(store, client, feature_level(store, kind), model, workers=workers, effort=effort, progress=progress)


def reopen(store: Store, kind: str) -> dict:
    return E.reopen(store, feature_level(store, kind))


def run_round(store: Store, client, kind: str = "guidance", *, batch_model: str = DEFAULT_MODEL, codebook_model: str = COLDSTART_MODEL, workers: int = 64, effort: str = "low",
              rounds: int = 5, encoder=None, log=None, progress=None, stop: Optional[threading.Event] = None) -> dict:
    r = E.run_round(store, client, feature_level(store, kind), batch_model=batch_model, codebook_model=codebook_model, workers=workers, effort=effort, rounds=rounds,
                    encoder=encoder, strong_model=COLDSTART_MODEL, log=log, progress=progress, stop=stop)
    return {"kind": kind, **r, "status": status(store, kind)}
