"""Running many calls: a thread pool with per-endpoint admission and a progress callback.

    from fx.core.llm import Client, run_many
    replies = run_many(client, jobs, model="openai/gpt-oss-20b", workers=16, stage="decompose",
                       progress=lambda done, total: ...)

`jobs` is a list of messages (a string or a message list each). Replies come back
in job order. A denied error on any job stops the pool, since every later job would
fail the same way; other errors are returned in place.

max_inflight caps how many calls sit at one endpoint at once. Without it a wide
worker pool floods a local vLLM queue past the client timeout and goodput collapses
while the GPU stays busy on requests nobody will read.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Sequence

from fx.core.llm.client import BudgetExceeded, Client, Messages, Reply
from fx.core.llm.registry import resolve

_sems: dict[str, threading.BoundedSemaphore] = {}
_sems_lock = threading.Lock()


def _sem(base_url: str, n: int) -> threading.BoundedSemaphore:
    with _sems_lock:
        if base_url not in _sems:
            _sems[base_url] = threading.BoundedSemaphore(n)
        return _sems[base_url]


def run_many(client: Client, jobs: Sequence[Messages], *, model: str, workers: int = 8, max_inflight: int = 64,
             progress: Optional[Callable[[int, int], None]] = None, stop_on_denied: bool = True, **kw) -> list[Optional[Reply]]:
    out: list[Optional[Reply]] = [None] * len(jobs)
    if not jobs:
        return out
    ep = resolve(model, client.base_url)
    sem = _sem(ep.base_url, max_inflight)
    stop = threading.Event()
    done = 0

    def one(i: int) -> tuple[int, Optional[Reply]]:
        if stop.is_set():
            return i, None
        with sem:
            if stop.is_set():
                return i, None
            r = client.complete(jobs[i], model=model, **kw)
        if stop_on_denied and r.error and r.error.startswith("denied"):
            stop.set()
        return i, r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, i) for i in range(len(jobs))]
        try:
            for f in as_completed(futs):
                i, r = f.result()
                out[i] = r
                done += 1
                if progress:
                    progress(done, len(jobs))
        except BudgetExceeded:
            stop.set()
            raise
    return out
