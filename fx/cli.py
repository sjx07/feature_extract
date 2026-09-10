"""fx: one command per stage over one store.

    fx llm probe --model openai/gpt-oss-20b          one call, prints the reply, tokens, cost, latency
    fx llm call  --model gpt-5.6-sol "prompt text"   a call with the ledger and cache
    fx llm spend                                     dollars by model from the store
    fx llm models                                    where each known model resolves

The workspace (store, logs, uploads, exports) comes from --workspace or FX_WORKSPACE, default
runs/dev; see fx.paths. Hosted keys
come from the environment: OPENAI_API_KEY, OPENROUTER_API_KEY, or FX_API_KEY with
--base-url for any other OpenAI-compatible server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


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
        p.add_argument("--model", default=os.environ.get("FX_MODEL", "deepseek/deepseek-v4-flash")); p.add_argument("--base-url", default=None)
        p.add_argument("--workers", type=int, default=128); p.add_argument("--redo", action="store_true")
        if name == "decompose":
            p.add_argument("--limit", type=int, default=0, help="first N prompts only (a pilot)")
            p.add_argument("--budget", type=float, default=float(os.environ.get("FX_BUDGET", "inf")))
            p.add_argument("--reasoning-on", action="store_true", help="leave the model's reasoning on (off by default for decomposition)")
    srv = sub.add_parser("serve"); srv.add_argument("--port", type=int, default=8780); srv.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)

    from .paths import Workspace
    from .store import Store
    ws = Workspace.from_env(a.workspace)
    if a.cmd == "llm" and a.sub == "models":
        from .llm.registry import PRICES, resolve, price
        for m in sorted(PRICES) + ["openai/gpt-oss-20b", "Qwen/Qwen2.5-7B-Instruct"]:
            ep = resolve(m)
            print(f"{m:36s} {ep.name:11s} {ep.base_url:40s} ${price(m, ep)[0]:.4f}/M in  ${price(m, ep)[1]:.4f}/M out")
        return 0
    store = Store(ws.store_path)
    if a.cmd == "import":
        import shutil
        from .corpus import import_path
        src = Path(a.path)
        if src.is_file():                                   # keep a verbatim copy beside the site's uploads
            kept = ws.upload_dir(a.name) / src.name
            if not kept.exists():
                shutil.copy2(src, kept)
        print(json.dumps(import_path(store, a.path, a.name, a.domain)))
        return 0
    if a.cmd == "preview":
        from .decompose import preview
        print(json.dumps(preview(store, a.corpus, a.model, a.workers, a.ids.split(",") if a.ids else None, a.redo, getattr(a, "limit", 0)), indent=1))
        return 0
    if a.cmd == "decompose":
        from . import jobs
        from .decompose import prompt_ids
        from .llm import Client
        ids = a.ids.split(",") if a.ids else None
        total = len(prompt_ids(store, a.corpus, ids, a.redo, a.limit))
        jid = jobs.start(store, ws, "decompose", a.corpus, a.model, {"workers": a.workers, "limit": a.limit, "redo": a.redo, "budget": a.budget if a.budget != float("inf") else None, "from": "cli"}, total)
        print(f"job {jid}: {total} prompts, log {ws.job_log(jid)}", flush=True)
        c = Client(store, budget=a.budget, base_url=a.base_url)
        status = jobs.run_decompose(store, ws, c, jid, a.corpus, model=a.model, workers=a.workers, ids=ids, redo=a.redo, limit=a.limit,
                                    reasoning="on" if a.reasoning_on else "off", echo=lambda line: print("  " + line, flush=True))
        print(f"job {jid} {status}")
        return 0 if status == "done" else 1
    if a.cmd == "serve":
        from .gui.server import serve
        serve(ws, a.host, a.port)
        return 0
    if a.sub == "spend":
        rows = store.spend_by_model()
        for r in rows:
            print(f"{r['model']:36s} calls {r['n']:6d} cached {r['cached'] or 0:6d} in {r['prompt_tokens'] or 0:10d} out {r['completion_tokens'] or 0:9d}  ${r['cost'] or 0:.4f}")
        print(f"total ${store.spent():.4f}")
        return 0
    from .llm import Client
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
