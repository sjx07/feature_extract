"""The loop: assign, judge, then reopen -> cluster -> name -> assign -> judge until settled. Each step is one logged line;
a re-run resumes from the store."""
from __future__ import annotations

import threading
from ..llm import Client
from ..store import Store
from typing import Callable, Optional
from .calls import Stopped
from .state import embed
from .assign import assign
from .judge import judge, reopen
from .group import regroup
from .name import candidates, name

# ---- the loop
def run_round(store: Store, client: Client, level, *, batch_model: str, codebook_model: str, workers: int = 128, effort: str = "low", rounds: int = 5,
              encoder=None, before: Optional[Callable[[Callable], None]] = None,
              log: Optional[Callable[[str, dict], None]] = None, progress=None, stop: Optional[threading.Event] = None) -> dict:
    """assign, judge, then reopen -> cluster -> name -> assign -> judge until settled. `before(step)` lets a level run its own
    first steps (collapse, cold start) through the same step logger; `level` may be a callable, resolved after `before`."""
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
        if before:
            before(step)
        lv = level() if callable(level) else level
        step("embed", lambda: embed(store, lv, enc=encoder))
        step("assign", lambda: assign(store, client, lv, batch_model, workers=workers, effort=effort, progress=progress, stop=stop))
        step("judge", lambda: judge(store, client, lv, batch_model, workers=workers, effort=effort, progress=progress))
        first = int(store.one("SELECT COALESCE(MAX(round), 0) r FROM feature WHERE codebook=?", (lv.codebook,))["r"]) + 1
        for rnd in range(first, first + rounds):
            r = step("reopen", lambda: reopen(store, lv))
            c = step("cluster", lambda: candidates(store, lv))
            if not c["clusters"] and r["reopened_misfits"] + r["reopened_split_members"] == 0:
                why = f"settled: every flag is standing and every open unit has had its look ({c['specific']} specific, {c['waiting']} waiting)"; break
            if c["clusters"]:
                step("name", lambda: name(store, client, lv, c["clusters"], rnd, codebook_model, workers=workers, effort=effort, progress=progress))
            if lv.prompt_group:
                step("group", lambda: regroup(store, client, lv, rnd, codebook_model, effort=effort, progress=progress))
            step("assign", lambda: assign(store, client, lv, batch_model, workers=workers, effort=effort, only_open=True, progress=progress, stop=stop))
            step("judge", lambda: judge(store, client, lv, batch_model, workers=workers, effort=effort, progress=progress))
    except Stopped:
        why = "stopped"
    return {"steps": steps, "stopped_because": why}
