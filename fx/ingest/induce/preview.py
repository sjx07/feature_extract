"""Calls and dollars for one library step, from the store's counts and the registry's prices."""
from __future__ import annotations

import math
from typing import Optional

from fx.data.corpus import corpus_id
from fx.core.llm.registry import DEFAULT_MODEL, price, resolve
from fx.core.store import Store
from fx.ingest.induce.codebook import BATCH, COLDSTART_MODEL, latest


def preview(store: Store, corpus: str, kind: str, step: str, model: Optional[str] = None, batch: int = BATCH) -> dict:
    """Calls and dollars for one step, from the store's counts and the registry's prices."""
    cid = corpus_id(store, corpus)
    n = int(store.one("SELECT COUNT(*) k FROM realization WHERE corpus=? AND kind=?", (cid, kind))["k"])
    chars = int(store.one("SELECT COALESCE(SUM(LENGTH(declaration)),0) c FROM realization WHERE corpus=? AND kind=?", (cid, kind))["c"])
    cbrow = latest(store, corpus, kind)
    nf = int(store.one("SELECT COUNT(*) k FROM feature WHERE codebook=? AND level='feature'", (cbrow["id"],))["k"]) if cbrow else 40
    ng = int(store.one("SELECT COUNT(*) k FROM feature WHERE codebook=? AND level='group'", (cbrow["id"],))["k"]) if cbrow else 8
    if step in ("collapse", "embed", "cluster"):
        return {"step": step, "model": None, "endpoint": "local", "calls": 0, "tokens_in": 0, "tokens_out": 0, "dollars": 0.0, "realizations": n}
    if step == "coldstart":
        model = model or COLDSTART_MODEL
        calls, tin, tout = 1, 1200 + (chars + 30 * n) // 3, 300 + 120 * max(nf, 30)
    elif step == "assign":
        model = model or DEFAULT_MODEL
        done = int(store.one("SELECT COUNT(*) k FROM membership WHERE kind='realization' AND codebook=?", (cbrow["id"],))["k"]) if cbrow else 0
        calls = math.ceil(max(n - done, 0) / batch)
        tin, tout = 900 + 40 * nf + 25 * batch, 20 * batch + 2000
    elif step == "judge":
        model = model or DEFAULT_MODEL
        calls, tin, tout = nf + ng, 1500, 2500
    elif step == "name":
        model = model or COLDSTART_MODEL
        open_ = int(store.one("SELECT COUNT(*) k FROM membership WHERE kind='realization' AND codebook=? AND node IS NULL AND (note IS NULL OR note != 'specific')", (cbrow["id"],))["k"]) if cbrow else n // 2
        calls, tin, tout = max(1, open_ // 8), 1500 + 40 * nf, 400        # about one cluster per eight open wordings, from the pilot
    else:
        raise ValueError(step)
    ep = resolve(model)
    pi, po = price(model, ep)
    return {"step": step, "model": model, "endpoint": ep.name, "calls": calls, "tokens_in": calls * tin, "tokens_out": calls * tout, "dollars": round(calls * (tin * pi + tout * po) / 1e6, 3), "realizations": n}
