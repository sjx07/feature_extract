"""The loop: assign, judge, then reopen -> cluster -> name -> assign -> judge until settled. Each step is one logged line;
a re-run resumes from the store."""
from __future__ import annotations

import threading
from fx.core.llm import Client
from fx.core.store import Store
from typing import Callable, Optional
from fx.ingest.loop.calls import Stopped
from fx.ingest.loop.state import embed
from fx.ingest.loop.assign import assign
from fx.ingest.loop.judge import judge, reopen
from fx.ingest.loop.join import join
from fx.ingest.loop.name import candidates, name

# ---- the loop
def run_round(store: Store, client: Client, level, *, batch_model: str, codebook_model: str, workers: int = 128, effort: str = "low", rounds: int = 5,
              encoder=None, strong_model: Optional[str] = None, before: Optional[Callable[[Callable], None]] = None,
              log: Optional[Callable[[str, dict], None]] = None, progress=None, stop: Optional[threading.Event] = None) -> dict:
    """assign, judge, then reopen -> cluster -> name -> assign -> judge until settled. `before(step)` lets a level run its own
    first steps (collapse, cold start) through the same step logger; `level` may be a callable, resolved after `before`."""
    stop = stop or threading.Event()
    steps: list[dict] = []

    def step(nm: str, fn) -> dict:
        r = fn()
        steps.append({"step": nm, **{k: v for k, v in r.items() if k not in ("notes", "clusters", "proposals")}})
        if log:
            log(nm, {k: v for k, v in r.items() if k not in ("clusters", "proposals")})
        if r.get("stopped") or stop.is_set():
            raise Stopped()
        return r

    why = f"{rounds} rounds done"
    try:
        if before:
            before(step)
        lv = level() if callable(level) else level
        step("embed", lambda: embed(store, lv, enc=encoder))
        step("assign", lambda: assign(store, client, lv, batch_model, workers=workers, effort=effort, progress=progress, stop=stop))
        step("judge", lambda: judge(store, client, lv, batch_model, workers=workers, effort=effort, progress=progress, strong_model=strong_model or codebook_model))
        first = int(store.one("SELECT COALESCE(MAX(round), 0) r FROM feature WHERE codebook=?", (lv.codebook,))["r"]) + 1
        for rnd in range(first, first + rounds):
            r = step("reopen", lambda: reopen(store, lv))
            c = step("cluster", lambda: candidates(store, lv))
            if not c["clusters"] and r["reopened_misfits"] + r["split_members_to_variants"] == 0:
                why = f"settled: every flag is standing and every open unit has had its look ({c['specific']} specific, {c['waiting']} waiting)"; break
            n = step("name", lambda: name(store, client, lv, c["clusters"], rnd, codebook_model, workers=workers, effort=effort, progress=progress)) if c["clusters"] else {"proposals": []}
            step("join", lambda: join(store, client, lv, n["proposals"], rnd, codebook_model, effort=effort, progress=progress))
            step("assign", lambda: assign(store, client, lv, batch_model, workers=workers, effort=effort, only_open=True, progress=progress, stop=stop))
            step("judge", lambda: judge(store, client, lv, batch_model, workers=workers, effort=effort, progress=progress, strong_model=strong_model or codebook_model))
    except Stopped:
        why = "stopped"
    return {"steps": steps, "stopped_because": why}
