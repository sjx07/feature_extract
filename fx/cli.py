"""fx: one command per stage over one store.

    fx llm probe --model openai/gpt-oss-20b          one call, prints the reply, tokens, cost, latency
    fx llm call  --model gpt-5.6-sol "prompt text"   a call with the ledger and cache
    fx llm spend                                     dollars by model from the store
    fx llm models                                    where each known model resolves

The store path comes from --store or FX_STORE (default runs/store.db). Hosted keys
come from the environment: OPENAI_API_KEY, OPENROUTER_API_KEY, or FX_API_KEY with
--base-url for any other OpenAI-compatible server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fx")
    ap.add_argument("--store", default=os.environ.get("FX_STORE", "runs/store.db"))
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
    a = ap.parse_args(argv)

    from .store import Store
    if a.sub == "models":
        from .llm.registry import PRICES, resolve, price
        for m in sorted(PRICES) + ["openai/gpt-oss-20b", "Qwen/Qwen2.5-7B-Instruct"]:
            ep = resolve(m)
            print(f"{m:36s} {ep.name:11s} {ep.base_url:40s} ${price(m, ep)[0]:.4f}/M in  ${price(m, ep)[1]:.4f}/M out")
        return 0
    store = Store(a.store)
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
