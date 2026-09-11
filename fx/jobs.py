"""One run of a stage, whoever started it: a `job` row in the store, a log file in the workspace,
progress into both. The CLI and the GUI call the same functions, so a run started from the
terminal shows on the site's job page, and a run started on the site has the same log file.

    jid = start(store, ws, "decompose", corpus, model, params, total)
    status = run_decompose(store, ws, client, jid, ...)      # updates the row as it goes, finishes it
"""
from __future__ import annotations

import json
import logging
import threading
import traceback
from typing import Optional

from .llm.registry import DEFAULT_MODEL
from .paths import Workspace
from .store import Store, now

log = logging.getLogger("fx.jobs")


def start(store: Store, ws: Workspace, kind: str, corpus: Optional[str], model: str, params: dict, total: int) -> int:
    jid = store.insert("job", {"kind": kind, "corpus": corpus, "model": model, "params": params, "status": "running", "total": total, "started": now(), "recent": []})
    with open(ws.job_log(jid), "a") as fh:
        fh.write(f"{now()} job {jid} {kind} corpus={corpus} model={model} total={total} params={json.dumps(params)}\n")
    log.info("job %d started: %s %s on %s, %d prompts", jid, kind, corpus, model, total)
    return jid


def finish(store: Store, ws: Workspace, jid: int, status: str, error: Optional[str] = None) -> None:
    with open(ws.job_log(jid), "a") as fh:                    # the log line first: a reader who sees the row finished finds the log complete
        fh.write(f"{now()} job {jid} {status}{': ' + error if error else ''}\n")
    with store.lock:
        store.con.execute("UPDATE job SET status=?, error=?, finished=? WHERE id=?", (status, error, now(), jid))
        store.con.commit()
    log.info("job %d %s%s", jid, status, f": {error}" if error else "")


def progress_writer(store: Store, ws: Workspace, jid: int, stop: threading.Event, echo=None):
    """A progress callback for a stage's run: updates the job row, appends a line per prompt to the
    job log, echoes to `echo` (the terminal) when given, and raises Stop once `stop` is set."""
    from .decompose import Stop

    def progress(done: int, total: int, info: dict) -> None:
        r = store.one("SELECT recent FROM job WHERE id=?", (jid,))
        recent = (json.loads(r["recent"] or "[]") + [{k: info.get(k) for k in ("id", "coverage", "n_atoms", "calls", "error")}])[-5:]
        with store.lock:
            store.con.execute("UPDATE job SET done=?, calls=calls+?, spent=spent+?, recent=? WHERE id=?",
                              (done, info.get("calls") or 0, info.get("cost") or 0.0, json.dumps(recent), jid))
            store.con.commit()
        line = f"{now()} {done}/{total} {info.get('id')} coverage={info.get('coverage')} atoms={info.get('n_atoms')} calls={info.get('calls')}" + (f" ERROR {info['error']}" if info.get("error") else "")
        with open(ws.job_log(jid), "a") as fh:
            fh.write(line + "\n")
        if echo:
            echo(line)
        if stop.is_set():
            raise Stop()
    return progress


def run_decompose(store: Store, ws: Workspace, client, jid: int, corpus: Optional[str], *, model: str, workers: int, ids=None, redo: bool = False,
                  limit: int = 0, stop: Optional[threading.Event] = None, echo=None) -> str:
    from . import decompose
    stop = stop or threading.Event()
    try:
        s = decompose.run(store, client, corpus, model=model, workers=workers, ids=ids, redo=redo, limit=limit,
                          progress=progress_writer(store, ws, jid, stop, echo), stop=stop)
        status = "stopped" if s["stopped"] else ("done" if not s["failed"] else "done_with_failures")
        finish(store, ws, jid, status)
    except Exception as e:
        with open(ws.job_log(jid), "a") as fh:
            fh.write(traceback.format_exc())
        finish(store, ws, jid, "failed", f"{type(e).__name__}: {str(e)[:300]}")
        status = "failed"
    return status


def run_library(store: Store, ws: Workspace, client, jid: int, corpus: str, kind: str, step: str, *, model: Optional[str] = None, workers: int = 16,
                version: Optional[int] = None, effort: str = "low", rounds: int = 3, codebook_model: Optional[str] = None,
                stop: Optional[threading.Event] = None, echo=None) -> str:
    """One library step as a job: coldstart, assign, judge, revise, or a whole round loop (collapse runs inline before coldstart and assign).
    `model` is the batch model (assign, judge); `codebook_model` the cold-start and revise model."""
    from . import library as L
    stop = stop or threading.Event()
    try:
        if step in ("coldstart", "assign"):
            L.collapse(store, corpus, kind)
        if step == "coldstart":
            r = L.coldstart(store, client, corpus, kind, model=model or L.COLDSTART_MODEL)
        elif step == "assign":
            r = L.assign(store, client, corpus, kind, model=model or DEFAULT_MODEL, version=version, workers=workers, effort=effort, progress=progress_writer(store, ws, jid, stop, echo), stop=stop)
        elif step == "judge":
            r = L.judge(store, client, corpus, kind, model=model or DEFAULT_MODEL, version=version, workers=workers, effort=effort, progress=progress_writer(store, ws, jid, stop, echo))
        elif step == "revise":
            r = L.revise(store, client, corpus, kind, model=model or L.COLDSTART_MODEL, version=version)
        elif step == "round":
            def log_step(name, res):
                with open(ws.job_log(jid), "a") as fh:
                    fh.write(f"{now()} step {name} result {json.dumps(res)}\n")
                if echo:
                    echo(f"step {name}: {json.dumps(res)[:300]}")
            r = L.run_round(store, client, corpus, kind, batch_model=model or DEFAULT_MODEL, codebook_model=codebook_model or L.COLDSTART_MODEL, workers=workers, effort=effort,
                            rounds=rounds, log=log_step, progress=progress_writer(store, ws, jid, stop, echo), stop=stop)
            r["stopped"] = r["stopped_because"] == "stopped"
        else:
            raise ValueError(step)
        with open(ws.job_log(jid), "a") as fh:
            fh.write(f"{now()} result {json.dumps(r)}\n")
        status = "stopped" if r.get("stopped") else "done"
        finish(store, ws, jid, status)
    except Exception as e:
        with open(ws.job_log(jid), "a") as fh:
            fh.write(traceback.format_exc())
        finish(store, ws, jid, "failed", f"{type(e).__name__}: {str(e)[:300]}")
        status = "failed"
    return status


def setup_logging(ws: Workspace, level: int = logging.INFO) -> None:
    """The site's log: to the workspace's logs/serve.log (rotated) and to the terminal."""
    from logging.handlers import RotatingFileHandler
    root = logging.getLogger()
    if any(getattr(h, "_fx", False) for h in root.handlers):
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(ws.logs / "serve.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt); fh._fx = True  # type: ignore[attr-defined]
    sh = logging.StreamHandler(); sh.setFormatter(fmt); sh._fx = True  # type: ignore[attr-defined]
    root.setLevel(level); root.addHandler(fh); root.addHandler(sh)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name); lg.handlers = []; lg.propagate = True
