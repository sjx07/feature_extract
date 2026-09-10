"""The localhost site: a FastAPI app over the store. Every response is a query over the store;
runs are jobs in a background thread that update their `job` row, which the page follows over
server-sent events.

    fx serve --port 8780         then open http://localhost:8780
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import decompose, jobs
from ..corpus import corpora, import_path, import_text
from ..llm import Client
from ..paths import Workspace
from ..store import Store, now

STATIC = Path(__file__).resolve().parent / "static"


def make_app(ws: Workspace, store: Optional[Store] = None) -> FastAPI:
    store = store or Store(ws.store_path)
    app = FastAPI(title="feature_extract")
    running: dict[int, threading.Event] = {}

    # ---- corpora and prompts
    @app.get("/api/corpora")
    def api_corpora():
        rows = corpora(store)
        for r in rows:
            d = store.one("SELECT COUNT(*) n, AVG(coverage) cov FROM decomp d JOIN prompt p ON p.id=d.prompt WHERE p.corpus=? AND d.status='done'", (r["id"],))
            r["decomposed"], r["coverage"] = int(d["n"] or 0), (round(d["cov"], 3) if d and d["cov"] is not None else None)
        return rows

    @app.post("/api/import")
    async def api_import(name: str = Form(...), domain: str = Form(""), file: Optional[UploadFile] = File(None), text: str = Form("")):
        if file is not None:
            data = await file.read()
            saved = ws.upload_dir(name) / Path(file.filename or "upload.txt").name        # kept verbatim, for provenance and re-import
            saved.write_bytes(data)
            return import_path(store, saved, name, domain or None)
        if text.strip():
            return import_text(store, text, name)
        raise HTTPException(400, "a file or a text is required")

    @app.get("/api/prompts")
    def api_prompts(corpus: str = "", status: str = "", limit: int = 500, offset: int = 0):
        sql = ("SELECT p.id, p.domain, p.system, p.task, SUBSTR(p.text, 1, 140) head, LENGTH(p.text) chars, c.name corpus, "
               "d.status, d.coverage, d.material_share, d.calls, "
               "(SELECT COUNT(*) FROM span s WHERE s.prompt=p.id AND s.kind='atom') n_atoms "
               "FROM prompt p JOIN corpus c ON c.id=p.corpus LEFT JOIN decomp d ON d.prompt=p.id WHERE 1=1")
        params: list = []
        if corpus:
            sql += " AND c.name=?"; params.append(corpus)
        if status == "done":
            sql += " AND d.status='done'"
        elif status == "todo":
            sql += " AND (d.status IS NULL OR d.status!='done')"
        sql += " ORDER BY p.rowid LIMIT ? OFFSET ?"; params += [limit, offset]
        rows = [dict(r) for r in store.rows(sql, params)]
        total = store.one("SELECT COUNT(*) n FROM prompt p JOIN corpus c ON c.id=p.corpus" + (" WHERE c.name=?" if corpus else ""), ([corpus] if corpus else []))["n"]
        return {"total": total, "prompts": rows}

    @app.get("/api/prompt/{pid:path}")
    def api_prompt(pid: str):
        p = store.one("SELECT p.*, c.name corpus FROM prompt p JOIN corpus c ON c.id=p.corpus WHERE p.id=?", (pid,))
        if not p:
            raise HTTPException(404, "no such prompt")
        d = store.one("SELECT * FROM decomp WHERE prompt=?", (pid,))
        spans = [dict(r) for r in store.rows("SELECT * FROM span WHERE prompt=? ORDER BY lo, hi DESC", (pid,))]
        by_span: dict[int, list] = {}
        for r in store.rows("SELECT * FROM reading WHERE prompt=? ORDER BY id", (pid,)):
            r = dict(r); r["domain_terms"] = json.loads(r["domain_terms"] or "[]")
            by_span.setdefault(r["span"], []).append(r)
        for a in spans:
            a["flags"] = json.loads(a["flags"] or "[]"); a["readings"] = by_span.get(a["id"], [])
        out = {**dict(p), "meta": json.loads(p["meta"] or "{}"), "decomp": dict(d) if d else None, "spans": spans}
        if d:
            out["decomp"]["flags"] = json.loads(d["flags"] or "[]"); out["decomp"]["failures"] = json.loads(d["failures"] or "[]")
        return out

    @app.get("/api/queues")
    def api_queues(corpus: str = "", min_coverage: float = 0.9):
        where, params = "", []
        if corpus:
            where, params = " AND c.name=?", [corpus]
        low = [dict(r) for r in store.rows("SELECT p.id, c.name corpus, d.coverage, (SELECT COUNT(*) FROM span s WHERE s.prompt=p.id AND s.kind='atom') n_atoms, SUBSTR(p.text,1,120) head "
                                           f"FROM decomp d JOIN prompt p ON p.id=d.prompt JOIN corpus c ON c.id=p.corpus WHERE d.status='done' AND d.coverage < ?{where} ORDER BY d.coverage LIMIT 200", [min_coverage] + params)]
        spans = lambda kind: [dict(r) for r in store.rows("SELECT s.prompt id, s.lo, s.hi, s.note, SUBSTR(p.text, s.lo+1, MIN(s.hi-s.lo, 160)) text FROM span s JOIN prompt p ON p.id=s.prompt JOIN corpus c ON c.id=p.corpus "
                                                          f"WHERE s.kind=?{where} ORDER BY s.hi-s.lo DESC LIMIT 300", [kind] + params)]
        failed = [dict(r) for r in store.rows("SELECT d.prompt id, d.error FROM decomp d JOIN prompt p ON p.id=d.prompt JOIN corpus c ON c.id=p.corpus WHERE d.status='failed'" + where, params)]
        return {"low_coverage": low, "gaps": spans("gap"), "unrefined": spans("unrefined"), "failed": failed}

    # ---- preview and jobs
    @app.get("/api/preview")
    def api_preview(corpus: str = "", model: str = "deepseek/deepseek-v4-flash", workers: int = 128, redo: bool = False, limit: int = 0):
        return decompose.preview(store, corpus or None, model, workers, redo=redo, limit=limit)

    @app.get("/api/jobs")
    def api_jobs():
        return [dict(r) | {"recent": json.loads(r["recent"] or "[]"), "params": json.loads(r["params"] or "{}")} for r in store.rows("SELECT * FROM job ORDER BY id DESC LIMIT 50")]

    @app.post("/api/jobs")
    def api_job_start(body: dict):
        corpus_name = body.get("corpus") or None
        model = body.get("model") or "deepseek/deepseek-v4-flash"
        workers = int(body.get("workers") or 128)
        limit = int(body.get("limit") or 0)
        redo = bool(body.get("redo"))
        budget = float(body["budget"]) if body.get("budget") not in (None, "", 0) else None
        ids = body.get("ids") or None
        reasoning = "off" if body.get("reasoning_off", True) else "on"
        total = len(decompose.prompt_ids(store, corpus_name, ids, redo, limit))
        params = {"workers": workers, "limit": limit, "redo": redo, "budget": budget, "ids": ids, "reasoning": reasoning, "from": "gui"}
        jid = jobs.start(store, ws, "decompose", corpus_name, model, params, total)
        stop = threading.Event()
        running[jid] = stop
        client = Client(store, budget=budget if budget is not None else float("inf"), base_url=body.get("base_url") or None)

        def work():
            jobs.run_decompose(store, ws, client, jid, corpus_name, model=model, workers=workers, ids=ids, redo=redo, limit=limit, reasoning=reasoning, stop=stop)
            running.pop(jid, None)

        threading.Thread(target=work, daemon=True).start()
        return {"id": jid, "total": total, "log": str(ws.job_log(jid))}

    @app.post("/api/jobs/{jid}/stop")
    def api_job_stop(jid: int):
        ev = running.get(jid)
        if ev:
            ev.set()
        return {"ok": bool(ev)}

    @app.get("/api/jobs/{jid}")
    def api_job(jid: int):
        r = store.one("SELECT * FROM job WHERE id=?", (jid,))
        if not r:
            raise HTTPException(404, "no such job")
        return dict(r) | {"recent": json.loads(r["recent"] or "[]"), "params": json.loads(r["params"] or "{}"), "elapsed": _elapsed(r), "log": str(ws.job_log(jid))}

    @app.get("/api/jobs/{jid}/log")
    def api_job_log(jid: int, tail: int = 200):
        p = ws.job_log(jid)
        lines = p.read_text().rstrip("\n").split("\n") if p.exists() else []
        return {"path": str(p), "lines": lines[-tail:]}

    @app.get("/api/jobs/{jid}/events")
    def api_job_events(jid: int):
        def gen():
            last = None
            while True:
                r = store.one("SELECT * FROM job WHERE id=?", (jid,))
                if not r:
                    yield "event: end\ndata: {}\n\n"; return
                d = dict(r) | {"recent": json.loads(r["recent"] or "[]"), "elapsed": _elapsed(r)}
                d.pop("params", None)
                s = json.dumps(d)
                if s != last:
                    yield f"data: {s}\n\n"; last = s
                if r["status"] != "running":
                    yield "event: end\ndata: {}\n\n"; return
                time.sleep(1.0)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/spend")
    def api_spend():
        return {"total": store.spent(), "by_model": store.spend_by_model(), "workspace": str(ws.root)}

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def _elapsed(r) -> Optional[float]:
    if not r["started"]:
        return None
    end = r["finished"] or now()
    fmt = "%Y-%m-%d %H:%M:%S"
    return round(time.mktime(time.strptime(end, fmt)) - time.mktime(time.strptime(r["started"], fmt)), 0)


def serve(ws: Workspace, host: str = "127.0.0.1", port: int = 8780) -> None:
    import uvicorn
    jobs.setup_logging(ws)
    logging.getLogger("fx").info("feature_extract on http://%s:%d  workspace %s", host, port, ws.root)
    uvicorn.run(make_app(ws), host=host, port=port, log_config=None)
