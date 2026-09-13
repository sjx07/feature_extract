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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from fx.ingest import decompose
from fx.core import jobs
from fx.ingest import induce as L
from fx.ingest import generalize as A
from fx.views import library as C
from fx.core import settings as S
from fx.views import ingest as I
from fx.data import history as H
from fx.data.corpus import corpora, import_path, import_text
from fx.core.llm.registry import DEFAULT_MODEL, LOCAL_URL, models
from fx.core.llm import Client
from fx.core.paths import Workspace
from fx.core.store import Store, now

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
    async def api_import(name: str = Form(...), domain: str = Form(""), file: Optional[UploadFile] = File(None), text: str = Form(""), path: str = Form("")):
        if path.strip():                                                                  # a path on the machine the site runs on
            p = Path(path.strip()).expanduser()
            if not p.exists():
                raise HTTPException(400, f"no such path on this machine: {p}")
            return import_path(store, p, name, domain or None)
        if file is not None:
            data = await file.read()
            saved = ws.upload_dir(name) / Path(file.filename or "upload.txt").name        # kept verbatim, for provenance and re-import
            saved.write_bytes(data)
            return import_path(store, saved, name, domain or None)
        if text.strip():
            return import_text(store, text, name, domain or None)
        raise HTTPException(400, "a path, a file or a text is required")

    @app.get("/api/models")
    def api_models():
        return {"default": DEFAULT_MODEL, "models": models(), "local_url": LOCAL_URL, "custom": os.environ.get("FX_MODELS", "")}

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
        unwrapped = [dict(r) | {"note": json.loads(r.pop("meta") or "{}").get("harvest", {}).get("wrapped", "")} for r in
                     (dict(x) for x in store.rows("SELECT p.id, p.meta, SUBSTR(p.text,1,120) text FROM prompt p JOIN corpus c ON c.id=p.corpus WHERE p.meta LIKE '%\"wrapped\":%'" + where + " LIMIT 300", params))]
        return {"low_coverage": low, "gaps": spans("gap"), "unrefined": spans("unrefined"), "failed": failed, "unwrapped": unwrapped}

    # ---- preview and jobs
    @app.get("/api/preview")
    def api_preview(corpus: str = "", model: str = DEFAULT_MODEL, workers: int = 512, redo: bool = False, limit: int = 0):
        return decompose.preview(store, corpus or None, model, workers, redo=redo, limit=limit)

    @app.get("/api/jobs")
    def api_jobs():
        from fx.core.jobs import reap
        reap(store)
        return [dict(r) | {"recent": json.loads(r["recent"] or "[]"), "params": json.loads(r["params"] or "{}")} for r in store.rows("SELECT * FROM job ORDER BY id DESC LIMIT 50")]

    @app.post("/api/jobs")
    def api_job_start(body: dict):
        corpus_name = body.get("corpus") or None
        model = body.get("model") or DEFAULT_MODEL
        workers = int(body.get("workers") or 128)
        limit = int(body.get("limit") or 0)
        redo = bool(body.get("redo"))
        budget = float(body["budget"]) if body.get("budget") not in (None, "", 0) else None
        ids = body.get("ids") or None
        total = len(decompose.prompt_ids(store, corpus_name, ids, redo, limit))
        params = {"workers": workers, "limit": limit, "redo": redo, "budget": budget, "ids": ids, "from": "gui"}
        jid = jobs.start(store, ws, "decompose", corpus_name, model, params, total)
        stop = threading.Event()
        running[jid] = stop
        client = Client(store, budget=budget if budget is not None else float("inf"), base_url=body.get("base_url") or None, max_connections=workers + 64)

        def work():
            jobs.run_decompose(store, ws, client, jid, corpus_name, model=model, workers=workers, ids=ids, redo=redo, limit=limit, stop=stop)
            running.pop(jid, None)

        threading.Thread(target=work, daemon=True).start()
        return {"id": jid, "total": total, "log": str(ws.job_log(jid))}

    # ---- stage 2: the library
    @app.get("/api/library")
    def api_library(corpus: str, kind: str = "guidance", version: Optional[int] = None):
        st = L.status(store, corpus, kind)
        cb = L.latest(store, corpus, kind, version)
        tree = L.groups(store, int(cb["id"])) if cb else []
        fl = L.flags(store, int(cb["id"])) if cb else []
        return st | {"codebook": cb, "groups": tree, "flags": fl, "leftover": L.leftovers(store, int(cb["id"]), corpus, kind)[:300] if cb else []}

    @app.get("/api/feature/{fid}")
    def api_feature(fid: int):
        f = store.one("SELECT * FROM feature WHERE id=?", (fid,))
        if not f:
            raise HTTPException(404, "no such feature")
        f = dict(f) | {"examples": json.loads(f["examples"] or "[]")}
        cb = dict(store.one("SELECT c.*, k.name corpus_name FROM codebook c JOIN corpus k ON k.id=c.corpus WHERE c.id=?", (f["codebook"],)))
        group = dict(store.one("SELECT * FROM feature WHERE id=?", (f["parent"],))) if f["parent"] else None
        ms = L.members(store, f["codebook"], fid, 500)
        readings = [dict(r) for r in store.rows("SELECT r.id, r.prompt, r.declaration, r.condition, s.lo, s.hi, SUBSTR(p.text, s.lo+1, MIN(s.hi-s.lo, 200)) text, r.realization FROM reading r JOIN span s ON s.id=r.span JOIN prompt p ON p.id=r.prompt "
                                                "WHERE r.realization IN (SELECT unit FROM membership WHERE kind='realization' AND codebook=? AND node=?) ORDER BY r.prompt LIMIT 300", (f["codebook"], fid))]
        fl = [x for x in L.flags(store, f["codebook"]) if x["feature"] == fid or x["other"] == fid]
        lineage = []
        prev = f["prev"]
        while prev:
            r = store.one("SELECT id, prev, name, codebook FROM feature WHERE id=?", (prev,))
            if not r:
                break
            lineage.append(dict(r)); prev = r["prev"]
        al = store.one("SELECT m.node global, m.note, g.name global_name FROM membership m LEFT JOIN feature g ON g.id=m.node WHERE m.kind='feature' AND m.unit=?", (fid,))
        return {"feature": f, "codebook": cb, "group": group, "members": ms, "readings": readings, "flags": fl, "lineage": lineage, "aligned": dict(al) if al else None}

    @app.get("/api/library/preview")
    def api_library_preview(corpus: str, kind: str = "guidance", step: str = "assign", model: str = ""):
        return L.preview(store, corpus, kind, step, model or None)

    @app.post("/api/library/jobs")
    def api_library_job(body: dict):
        corpus_name, kind, step = body["corpus"], body.get("kind") or "guidance", body.get("step") or "assign"
        model = body.get("model") or (L.COLDSTART_MODEL if step == "coldstart" else DEFAULT_MODEL)
        workers = int(body.get("workers") or 128)
        version = int(body["version"]) if body.get("version") else None
        effort = body.get("effort") or "low"
        rounds, codebook_model = int(body.get("rounds") or 5), body.get("codebook_model") or None
        params = {"kind": kind, "version": version, "workers": workers, "effort": effort, "rounds": rounds, "codebook_model": codebook_model, "from": "gui"}
        jid = jobs.start(store, ws, f"library:{step}", corpus_name, model, params, 0)
        stop = threading.Event()
        running[jid] = stop
        budget = float(body["budget"]) if body.get("budget") not in (None, "", 0) else float("inf")
        client = Client(store, base_url=body.get("base_url") or None, max_connections=workers + 64, budget=budget)

        def work():
            jobs.run_library(store, ws, client, jid, corpus_name, kind, step, model=model, workers=workers, version=version, effort=effort, rounds=rounds, codebook_model=codebook_model, stop=stop)
            running.pop(jid, None)

        threading.Thread(target=work, daemon=True).start()
        return {"id": jid, "log": str(ws.job_log(jid))}

    # ---- stage 3: the seed library
    @app.get("/api/seed")
    def api_seed(kind: str = "guidance"):
        cb = A.seed_codebook(store, kind)
        fl = [dict(r) for r in store.rows("SELECT f.*, x.name global_name, y.name member_name FROM flag f JOIN feature x ON x.id=f.feature LEFT JOIN feature y ON y.id=f.other WHERE f.codebook=?", (cb,))]
        return {"status": A.status(store, kind), "groups": A.globals_(store, kind), "open": A.open_cards(store, kind), "flags": fl, "libraries": A.libraries(store, kind)}   # counts under status; the lists keep their names

    @app.post("/api/align/jobs")
    def api_align_job(body: dict):
        kind, step = body.get("kind") or "guidance", body.get("step") or "round"
        model = body.get("model") or DEFAULT_MODEL
        workers = int(body.get("workers") or 64)
        params = {"kind": kind, "workers": workers, "effort": body.get("effort") or "low", "rounds": int(body.get("rounds") or 5), "codebook_model": body.get("codebook_model") or None, "from": "gui"}
        jid = jobs.start(store, ws, f"align:{step}", "seed", model, params, 0)
        stop = threading.Event()
        running[jid] = stop
        budget = float(body["budget"]) if body.get("budget") not in (None, "", 0) else float("inf")
        client = Client(store, base_url=body.get("base_url") or None, max_connections=workers + 64, budget=budget)

        def work():
            jobs.run_align(store, ws, client, jid, kind, step, model=model, codebook_model=params["codebook_model"], workers=workers, effort=params["effort"], rounds=params["rounds"], stop=stop)
            running.pop(jid, None)

        threading.Thread(target=work, daemon=True).start()
        return {"id": jid, "log": str(ws.job_log(jid))}

    # ---- the cube: the Library side, readings × prompt fields × the seed's hierarchy
    @app.get("/api/cube")
    def api_cube(request: Request, kind: str = "guidance", view: str = "features", limit: int = 300):
        f = C.parse_filters(dict(request.query_params))
        return C.prompts(store, kind, f, limit) if view == "prompts" else C.slice(store, kind, f)

    # ---- ingest: corpora kept, profiles, one run through the stages
    @app.get("/api/ingest")
    def api_ingest(kind: str = "guidance"):
        js = [dict(r) | {"params": json.loads(r["params"] or "{}")} for r in store.rows("SELECT id, kind, corpus, model, status, done, total, spent, started, finished, params FROM job ORDER BY id DESC LIMIT 40")]
        live = {j["id"]: j["live"] for j in I.running_jobs(store, ws)}
        for j in js:
            if j["status"] == "running" and not live.get(j["id"], True):
                j["status"] = "stale"
        from fx.data.corpus import imports
        from fx.data.tags import fields
        return {"corpora": I.corpora(store, kind, ws), "profiles": I.profiles(store), "jobs": js, "imports": imports(store, limit=30), "fields": fields(store), "default_model": DEFAULT_MODEL, "models": models()}

    @app.post("/api/jobs/{jid}/close")
    def api_job_close(jid: int):
        """A row left running by a process that died: closed by hand from the jobs page."""
        if jid in running:
            raise HTTPException(409, "this job is running in this server; stop it instead")
        return I.close_job(store, jid)

    @app.get("/api/ingest/estimate")
    def api_ingest_estimate(profile: str = "default", corpora: str = ""):
        try:
            prof = I.profile(store, profile)
        except KeyError:
            raise HTTPException(404, "no such profile")
        kind = prof.get("kind") or "guidance"
        names = [n for n in corpora.split(",") if n] or [c["name"] for c in I.corpora(store, "guidance" if kind == "both" else kind, ws) if c["pending"]]
        est = {n: I.estimate(store, n, prof, kind) for n in names}
        return {"profile": profile, "kind": kind, "corpora": est, "total": round(sum(e["total"] for e in est.values()), 2)}

    @app.post("/api/corpus/{name}/rename")
    def api_corpus_rename(name: str, body: dict):
        try:
            return I.rename(store, name, str(body.get("name") or ""))
        except KeyError:
            raise HTTPException(404, "no such corpus")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/corpus/{name}/retag")
    def api_corpus_retag(name: str, body: dict):
        try:
            return I.retag(store, name, body.get("domain"), body.get("tags") or None)
        except KeyError:
            raise HTTPException(404, "no such corpus")

    @app.delete("/api/corpus/{name}")
    def api_corpus_delete(name: str):
        if name in {j for j in running_corpora()}:
            raise HTTPException(409, "a job is running on this corpus; stop it first")
        try:
            return I.delete(store, name)
        except KeyError:
            raise HTTPException(404, "no such corpus")

    def running_corpora():
        return [r["corpus"] for r in store.rows("SELECT corpus FROM job WHERE status='running'")]

    @app.post("/api/prompts/delete")
    def api_prompts_delete(body: dict):
        return I.delete_prompts(store, [str(x) for x in (body.get("ids") or [])])

    @app.get("/api/profiles")
    def api_profiles():
        return I.profiles(store)

    @app.post("/api/profiles")
    def api_profile_save(body: dict):
        try:
            return I.save_profile(store, str(body.get("name") or ""), body.get("params") or {})
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/profiles/{name}")
    def api_profile_delete(name: str):
        I.delete_profile(store, name); return {"ok": True}

    @app.post("/api/ingest/run")
    def api_ingest_run(body: dict):
        """One job per corpus, run one after another in one thread under one profile; 'pending' means every corpus with a stage to do."""
        try:
            prof = I.profile(store, body.get("profile") or "default")
        except KeyError:
            raise HTTPException(404, "no such profile")
        kind = body.get("kind") or prof.get("kind") or "guidance"
        names = body.get("corpora") or []
        if names == "pending" or body.get("pending"):
            names = [c["name"] for c in I.corpora(store, "guidance" if kind == "both" else kind, ws) if c["pending"] and not c["running"]]
        names = [n for n in names if n not in running_corpora()]
        if not names:
            raise HTTPException(400, "nothing to run")
        budget = float(prof.get("budget") or 0) or float("inf")
        jids = []
        for n in names:
            jids.append(jobs.start(store, ws, "profile", n, prof.get("batch_model") or DEFAULT_MODEL, {"profile": body.get("profile") or "default", "kind": kind, "stage": "queued", "from": "gui"} | {k: prof.get(k) for k in ("decompose_model", "codebook_model", "batch_model", "workers", "effort")}, 0))
        stop = threading.Event()
        for jid in jids:
            running[jid] = stop

        def work():
            for n, jid in zip(names, jids):
                if stop.is_set():
                    jobs.finish(store, ws, jid, "stopped"); running.pop(jid, None); continue
                client = Client(store, budget=budget, base_url=prof.get("base_url") or None, max_connections=int(prof.get("workers") or 128) + 64)
                try:
                    H.checkpoint(store, ws, n, job=jid, note="before the run"); H.checkpoint(store, ws, "seed", job=jid, note="before the run") if store.one("SELECT 1 FROM codebook WHERE scope='seed'") else None
                except Exception as e:  # noqa: BLE001  a checkpoint failure must not stop the run
                    logging.getLogger("fx").warning("checkpoint before job %d failed: %s", jid, e)
                jobs.run_profile(store, ws, client, jid, n, prof, kind=kind, stop=stop)
                running.pop(jid, None)

        threading.Thread(target=work, daemon=True).start()
        return {"ids": jids, "corpora": names}

    # ---- history: checkpoints per corpus, restore, diff, branch
    @app.get("/api/history")
    def api_history(corpus: str = ""):
        return {"checkpoints": H.checkpoints(store, corpus or None), "objects": str(H.objects_dir(ws)), "workspace": str(ws.root)}

    @app.post("/api/history/checkpoint")
    def api_history_checkpoint(body: dict):
        try:
            return H.checkpoint(store, ws, str(body.get("corpus") or ""), note=body.get("note") or "by hand")
        except KeyError:
            raise HTTPException(404, "no such corpus")

    @app.get("/api/history/{cid}/diff")
    def api_history_diff(cid: int):
        try:
            return H.diff(store, cid, ws)
        except KeyError:
            raise HTTPException(404, "no such checkpoint")

    @app.post("/api/history/{cid}/restore")
    def api_history_restore(cid: int):
        live = {j["corpus"] for j in I.running_jobs(store, ws) if j["live"]}
        try:
            r = H.restore(store, ws, cid, live_corpora=live)
        except KeyError:
            raise HTTPException(404, "no such checkpoint")
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        C._cache.clear()
        return r

    @app.post("/api/history/{cid}/branch")
    def api_history_branch(cid: int, body: dict):
        try:
            return H.branch(store, ws, cid, str(body.get("name") or ""))
        except KeyError:
            raise HTTPException(404, "no such checkpoint")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/jobs/{jid}/stages")
    def api_job_stages(jid: int):
        r = store.one("SELECT corpus, params FROM job WHERE id=?", (jid,))
        if not r:
            raise HTTPException(404, "no such job")
        k = json.loads(r["params"] or "{}").get("kind") or "guidance"
        return I.stages(store, r["corpus"], "guidance" if k == "both" else k)

    @app.get("/api/cube/node/{fid}")
    def api_cube_node(request: Request, fid: int, kind: str = "guidance"):
        d = C.node(store, kind, C.parse_filters(dict(request.query_params)), fid)
        if not d:
            raise HTTPException(404, "no such feature in this kind's libraries")
        return d

    # ---- settings: keys and endpoints by reference; a key is written blind and never read back
    @app.get("/api/settings")
    def api_settings():
        return S.status() | {"default_model": DEFAULT_MODEL, "models": models()}

    @app.post("/api/settings/key")
    def api_settings_key(body: dict):
        name, value = str(body.get("name") or "").strip(), str(body.get("value") or "")
        if name not in S.KEYS and name not in S.SETTINGS:
            raise HTTPException(400, f"not a known key or setting: {name}")
        try:
            return S.save(name, value.strip())
        except (ValueError, OSError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/settings/probe")
    def api_settings_probe(body: dict):
        return S.probe(body.get("endpoint") or None, body.get("base_url") or None)

    @app.post("/api/jobs/{jid}/stop")
    def api_job_stop(jid: int):
        ev = running.get(jid)
        if ev:
            ev.set()
            with store.lock:
                store.con.execute("UPDATE job SET status='stopping' WHERE id=? AND status='running'", (jid,)); store.con.commit()
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

    @app.middleware("http")
    async def no_cache(request, call_next):
        """The script, style and index are re-read on every load, so a restart on new code is never hidden by the browser's cache."""
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp

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
