"""Calls and dollars for one library step, from the store's counts and the registry's prices."""
from __future__ import annotations

import math
from typing import Optional

from ..corpus import corpus_id
from ..llm.registry import DEFAULT_MODEL, price, resolve
from ..store import Store
from .codebook import BATCH, COLDSTART_MODEL, latest


def preview(store: Store, corpus: str, kind: str, step: str, model: Optional[str] = None, batch: int = BATCH) -> dict:
    """Calls and dollars for one step, from the store's counts and the registry's prices."""
    cid = corpus_id(store, corpus)
    n = int(store.one("SELECT COUNT(*) k FROM realization WHERE corpus=? AND kind=?", (cid, kind))["k"])
    chars = int(store.one("SELECT COALESCE(SUM(LENGTH(declaration)),0) c FROM realization WHERE corpus=? AND kind=?", (cid, kind))["c"])
    cbrow = latest(store, corpus, kind)
    nf = int(store.one("SELECT COUNT(*) k FROM feature WHERE codebook=? AND level='feature'", (cbrow["id"],))["k"]) if cbrow else 40
    ng = int(store.one("SELECT COUNT(*) k FROM feature WHERE codebook=? AND level='group'", (cbrow["id"],))["k"]) if cbrow else 8
    if step in ("coldstart", "revise"):
        model = model or COLDSTART_MODEL
        calls, tin, tout = 1, 1200 + (chars + 30 * n) // 3, 300 + 120 * max(nf, 30)
    elif step == "assign":
        model = model or DEFAULT_MODEL
        done = int(store.one("SELECT COUNT(*) k FROM assignment WHERE codebook=?", (cbrow["id"],))["k"]) if cbrow else 0
        calls = math.ceil(max(n - done, 0) / batch)
        tin, tout = 900 + 40 * nf + 25 * batch, 20 * batch + 2000
    elif step == "judge":
        model = model or DEFAULT_MODEL
        calls, tin, tout = nf + ng, 1500, 2500
    else:
        raise ValueError(step)
    ep = resolve(model)
    pi, po = price(model, ep)
    return {"step": step, "model": model, "endpoint": ep.name, "calls": calls, "tokens_in": calls * tin, "tokens_out": calls * tout, "dollars": round(calls * (tin * pi + tout * po) / 1e6, 3), "realizations": n}
