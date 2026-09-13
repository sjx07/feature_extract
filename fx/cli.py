"""fx: one command per stage over one store.

    fx llm probe --model openai/gpt-oss-20b          one call, prints the reply, tokens, cost, latency
    fx llm call  --model gpt-5.6-sol "prompt text"   a call with the ledger and cache
    fx llm spend                                     dollars by model from the store
    fx llm models                                    where each known model resolves

The workspace (store, logs, uploads, exports) comes from --workspace or FX_WORKSPACE, default
runs/dev; see fx.core.paths. Hosted keys
come from the environment: OPENAI_API_KEY, OPENROUTER_API_KEY, or FX_API_KEY with
--base-url for any other OpenAI-compatible server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fx.core.llm.registry import DEFAULT_MODEL
from fx.ingest.induce.codebook import COLDSTART_MODEL as L_COLDSTART


def _detach() -> None:
    """A model run keeps going when the terminal that started it goes away: hang-ups are ignored and the job log is the
    record; a dead stdout only silences the echo."""
    import signal
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except (AttributeError, ValueError, OSError):
        pass


def _echo(line: str) -> None:
    try:
        print("  " + line, flush=True)
    except (OSError, ValueError):
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fx")
    ap.add_argument("--workspace", "-w", default=None, help="runs/<name>; default FX_WORKSPACE or runs/dev")
    sub = ap.add_subparsers(dest="cmd", required=True)
    llm = sub.add_parser("llm").add_subparsers(dest="sub", required=True)
    for name in ("probe", "call"):
        p = llm.add_parser(name)
        p.add_argument("--model", default=os.environ.get("FX_MODEL", "openai/gpt-oss-20b"))
        p.add_argument("--base-url", default=None)
        p.add_argument("--max-tokens", type=int, default=512)
        p.add_argument("--budget", type=float, default=float(os.environ.get("FX_BUDGET", "inf")))
        p.add_argument("--no-cache", action="store_true")
        p.add_argument("--stream", action="store_true")
        p.add_argument("--reasoning-off", action="store_true", help="send extra_body reasoning.enabled=false (OpenRouter models)")
        if name == "call":
            p.add_argument("prompt")
    llm.add_parser("spend")
    llm.add_parser("models")
    imp = sub.add_parser("import", help="a FACET prompts.jsonl, a folder or zip of text files, or one text file")
    imp.add_argument("path"); imp.add_argument("--name", required=True); imp.add_argument("--domain", default=None, help="keep only this domain from a FACET jsonl")
    for name in ("preview", "decompose"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", default=None); p.add_argument("--ids", default=None, help="comma-separated prompt ids")
        p.add_argument("--model", default=DEFAULT_MODEL); p.add_argument("--base-url", default=None)
        p.add_argument("--workers", type=int, default=512, help="calls in flight across the run"); p.add_argument("--redo", action="store_true")
        p.add_argument("--limit", type=int, default=0, help="first N prompts only (a pilot)")
        if name == "decompose":
            p.add_argument("--budget", type=float, default=float(os.environ.get("FX_BUDGET", "inf")))
    lib = sub.add_parser("library", help="stage 2: the feature library of a corpus").add_subparsers(dest="sub", required=True)
    for name in ("round", "collapse", "coldstart", "assign", "judge", "reopen", "cluster", "name", "status", "preview", "relook"):
        p = lib.add_parser(name)
        p.add_argument("--corpus", required=True); p.add_argument("--kind", default="guidance", choices=("guidance", "material"))
        p.add_argument("--model", default=None, help="default: the decomposition model for assign and judge; --codebook-model for coldstart and name")
        p.add_argument("--version", type=int, default=None, help="codebook version (default: latest)"); p.add_argument("--workers", type=int, default=128); p.add_argument("--effort", default="low", choices=("low", "default"), help="reasoning effort for assign and judge batches")
        p.add_argument("--base-url", default=None, help="any OpenAI-compatible server for this step's model (e.g. a second local vLLM)")
        p.add_argument("--budget", type=float, default=float(os.environ.get("FX_BUDGET", "inf")), help="dollars this run may spend")
        if name == "round":
            p.add_argument("--rounds", type=int, default=5)
        p.add_argument("--codebook-model", default=None, help=f"cold start and naming model (default {L_COLDSTART})")
        if name == "preview":
            p.add_argument("--step", default="assign", choices=("coldstart", "assign", "judge", "name"))
        if name == "coldstart":
            p.add_argument("--fresh", action="store_true", help="call the model again even if this corpus's cold start is cached (a rebuild, not a replay)")
    al = sub.add_parser("align", help="stage 3: the seed library, per-corpus features aligned into global features").add_subparsers(dest="sub", required=True)
    for name in ("round", "embed", "assign", "judge", "reopen", "cluster", "name", "status", "regroup", "reset"):
        p = al.add_parser(name)
        p.add_argument("--kind", default="guidance", choices=("guidance", "material"))
        p.add_argument("--model", default=None, help="batch model for assign and judge (default: the decomposition model)")
        p.add_argument("--codebook-model", default=None, help=f"naming model (default {L_COLDSTART})")
        p.add_argument("--workers", type=int, default=64); p.add_argument("--effort", default="low", choices=("low", "default"))
        p.add_argument("--rounds", type=int, default=5)
        p.add_argument("--base-url", default=None); p.add_argument("--budget", type=float, default=float(os.environ.get("FX_BUDGET", "inf")))
    hist = sub.add_parser("history", help="git for the store, per corpus: checkpoints as shared blobs, restore, diff, branch").add_subparsers(dest="sub", required=True)
    for name in ("checkpoint", "list", "restore", "diff", "branch"):
        p = hist.add_parser(name)
        p.add_argument("--corpus", default=None); p.add_argument("--id", type=int, default=None, help="a checkpoint id (restore, diff, branch)")
        p.add_argument("--name", default=None, help="the branch's workspace name (branch)"); p.add_argument("--note", default=None)
    st = sub.add_parser("store", help="the store's schema version; migrate = open it, which runs the steps").add_subparsers(dest="sub", required=True)
    st.add_parser("version"); st.add_parser("migrate")
    srv = sub.add_parser("serve"); srv.add_argument("--port", type=int, default=8780); srv.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)

    from fx.core.paths import Workspace
    from fx.core.store import Store
    ws = Workspace.from_env(a.workspace)
    if a.cmd == "llm" and a.sub == "models":
        from fx.core.llm.registry import models
        for r in models():
            print(f"{r['model']:36s} {r['endpoint']:11s} ${r['price_in']:.4f}/M in  ${r['price_out']:.4f}/M out{'  (default)' if r['default'] else ''}")
        return 0
    store = Store(ws.store_path)
    if a.cmd == "import":
        import shutil
        from fx.data.corpus import import_path
        src = Path(a.path)
        if src.is_file():                                   # keep a verbatim copy beside the site's uploads
            kept = ws.upload_dir(a.name) / src.name
            if not kept.exists():
                shutil.copy2(src, kept)
        print(json.dumps(import_path(store, a.path, a.name, a.domain)))
        return 0
    if a.cmd == "preview":
        from fx.ingest.decompose import preview
        print(json.dumps(preview(store, a.corpus, a.model, a.workers, a.ids.split(",") if a.ids else None, a.redo, getattr(a, "limit", 0)), indent=1))
        return 0
    if a.cmd == "decompose":
        from fx.core import jobs
        from fx.ingest.decompose import prompt_ids
        from fx.core.llm import Client
        ids = a.ids.split(",") if a.ids else None
        total = len(prompt_ids(store, a.corpus, ids, a.redo, a.limit))
        _detach()
        jid = jobs.start(store, ws, "decompose", a.corpus, a.model, {"workers": a.workers, "limit": a.limit, "redo": a.redo, "budget": a.budget if a.budget != float("inf") else None, "from": "cli"}, total)
        print(f"job {jid}: {total} prompts, log {ws.job_log(jid)}", flush=True)
        c = Client(store, budget=a.budget, base_url=a.base_url, max_connections=a.workers + 64)
        status = jobs.run_decompose(store, ws, c, jid, a.corpus, model=a.model, workers=a.workers, ids=ids, redo=a.redo, limit=a.limit,
                                    echo=_echo)
        print(f"job {jid} {status}")
        return 0 if status == "done" else 1
    if a.cmd == "library":
        from fx.ingest import induce as L
        if a.sub == "collapse":
            print(json.dumps(L.collapse(store, a.corpus, a.kind))); return 0
        if a.sub == "status":
            print(json.dumps(L.status(store, a.corpus, a.kind), indent=1)); return 0
        if a.sub == "relook":
            print(json.dumps(L.relook(store, a.corpus, a.kind, a.version))); return 0
        if a.sub == "preview":
            print(json.dumps(L.preview(store, a.corpus, a.kind, a.step, a.model), indent=1)); return 0
        from fx.core.jobs import run_library, setup_logging, start
        from fx.core.llm import Client
        setup_logging(ws)
        model = a.model or (L.COLDSTART_MODEL if a.sub == "coldstart" else DEFAULT_MODEL)
        rounds, codebook_model = getattr(a, "rounds", 5), getattr(a, "codebook_model", None)
        _detach()
        jid = start(store, ws, f"library:{a.sub}", a.corpus, model, {"kind": a.kind, "version": a.version, "workers": a.workers, "effort": a.effort, "rounds": rounds, "codebook_model": codebook_model, "budget": a.budget if a.budget != float("inf") else None, "from": "cli"}, 0)
        print(f"job {jid}: {a.sub} {a.kind} on {a.corpus}, log {ws.job_log(jid)}")
        status = run_library(store, ws, Client(store, base_url=a.base_url, max_connections=a.workers + 64, budget=a.budget), jid, a.corpus, a.kind, a.sub, model=model, workers=a.workers, version=a.version, effort=a.effort, rounds=rounds, codebook_model=codebook_model, fresh=getattr(a, "fresh", False), echo=_echo)
        print(status); print(open(ws.job_log(jid)).read().strip().split("\n")[-2][:600] if status == "done" else "")
        return 0 if status == "done" else 1
    if a.cmd == "align":
        from fx.ingest import generalize as A
        if a.sub == "status":
            print(json.dumps(A.status(store, a.kind), indent=1)); return 0
        if a.sub == "regroup":
            print(json.dumps(A.regroup_by_aspect(store, a.kind))); return 0
        if a.sub == "reset":
            live = store.one("SELECT id FROM job WHERE kind LIKE 'align:%' AND status='running'")
            print(json.dumps(A.reset(store, a.kind) | ({"warning": f"job {live['id']} shows as running; stop it first if it is"} if live else {}))); return 0
        from fx.core.jobs import run_align, setup_logging, start
        from fx.core.llm import Client
        setup_logging(ws)
        model = a.model or DEFAULT_MODEL
        _detach()
        jid = start(store, ws, f"align:{a.sub}", "seed", model, {"kind": a.kind, "workers": a.workers, "effort": a.effort, "rounds": a.rounds, "codebook_model": a.codebook_model, "budget": a.budget if a.budget != float("inf") else None, "from": "cli"}, 0)
        print(f"job {jid}: align {a.sub} {a.kind}, log {ws.job_log(jid)}")
        status = run_align(store, ws, Client(store, base_url=a.base_url, max_connections=a.workers + 64, budget=a.budget), jid, a.kind, a.sub, model=model, codebook_model=a.codebook_model,
                           workers=a.workers, effort=a.effort, rounds=a.rounds, echo=_echo)
        print(status)
        return 0 if status == "done" else 1
    if a.cmd == "store":
        from fx.core import migrations
        print(f"{ws.store_path}: schema version {store.version}, the code knows {migrations.CURRENT}" + (f"; applied now: {store.applied}" if store.applied else ""))
        return 0
    if a.cmd == "history":
        from fx.data import history as H
        if a.sub == "checkpoint":
            print(json.dumps(H.checkpoint(store, ws, a.corpus or "seed", note=a.note or "by hand"), indent=1))
        elif a.sub == "list":
            for c in H.checkpoints(store, a.corpus):
                print(f"#{c['id']:<4} {c['corpus']:<22} {c['at']}  before job {c['job'] or '-':<5} {c['note'] or '':<16} {json.dumps(c['counts'])}")
        elif a.sub == "diff":
            print(json.dumps(H.diff(store, a.id, ws), indent=1))
        elif a.sub == "restore":
            print(json.dumps(H.restore(store, ws, a.id), indent=1))
        elif a.sub == "branch":
            print(json.dumps(H.branch(store, ws, a.id, a.name or f"branch-{a.id}"), indent=1))
        return 0
    if a.cmd == "serve":
        from fx.gui.server import serve
        serve(ws, a.host, a.port)
        return 0
    if a.sub == "spend":
        rows = store.spend_by_model()
        for r in rows:
            print(f"{r['model']:36s} calls {r['n']:6d} cached {r['cached'] or 0:6d} in {r['prompt_tokens'] or 0:10d} out {r['completion_tokens'] or 0:9d}  ${r['cost'] or 0:.4f}")
        print(f"total ${store.spent():.4f}")
        return 0
    from fx.core.llm import Client
    c = Client(store, budget=a.budget, base_url=a.base_url)
    prompt = a.prompt if a.sub == "call" else "Reply with the single word: ready."
    extra = {"reasoning": {"enabled": False}} if a.reasoning_off else None
    r = c.complete(prompt, model=a.model, max_tokens=a.max_tokens, cache=not a.no_cache, stream=a.stream, extra_body=extra, stage="cli",
                   on_chunk=(lambda s: (sys.stdout.write(s), sys.stdout.flush())) if a.stream else None)
    if a.stream:
        print()
    else:
        print(r.text)
    print(json.dumps({"model": r.model, "usage": r.usage, "cost": round(r.cost, 6), "latency": r.latency, "finish": r.finish_reason,
                      "cached": r.cached, "error": r.error, "call_id": r.call_id, "spent": round(store.spent(), 4)}), file=sys.stderr)
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
