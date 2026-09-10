# Stage 0: LLM client, ledger, cache, pool

Branch `stage0-llm` against `main`. Seven files, 763 lines, 16 tests.

## What it gives every later stage

- `fx.store.Store`: one SQLite file (WAL) with the `call` table: model, endpoint, prompt hash, tokens, cost, latency, finish reason, error, cached flag, reply. `spent()` and `spend_by_model()` read the ledger from it; later stages add tables with `migrate`.
- `fx.llm.Client.complete(messages, model=...)` returns a `Reply` and never raises for a model-side failure. It probes hosted parameter shapes once per model (`max_completion_tokens`, temperature unsupported), retries empty replies and transport failures with backoff, reports denials and unreachable endpoints as errors without logging a paid call, streams long generations with a chunk callback, serves repeats from the cache at no cost, prices each paid call from the registry, and refuses to start a call once the budget is spent (`BudgetExceeded`).
- `fx.llm.run_many`: thread pool with per-endpoint admission, ordered results, progress callback, stop on denial. `with_fallback`: chained model settings, such as reasoning off on a fallback.
- `fx.llm.registry`: `gpt-*` to OpenAI, `vendor/model` to OpenRouter, anything else to the local vLLM server; `FX_MODELS` JSON adds endpoints and prices without code; unknown hosted models priced at the top rate.
- `fx llm probe | call | spend | models`.

## Carried over from FACET, and what changed

The behaviours come from `facet/llm/call.py`, `facet/llm/ledger.py`, `facet/llm/cache.py`, and the cold start's streaming call, each of which was added after a failure on the v5 runs. Changes: one client instead of three factories and a worker seam; the cache and the ledger live in the store instead of jsonl files; the budget is enforced inside the client rather than by a wrapper; denials and unreachable endpoints are classified in one place; streaming is a flag on the same call.

## How to check

```
cd ~/Documents/feature_extract
python -m pytest -q                                         # 16 tests, scripted fake server, no network
PYTHONPATH=. python -m fx.cli llm models
PYTHONPATH=. python -m fx.cli llm probe --model openai/gpt-oss-20b          # local vLLM on :8000
PYTHONPATH=. python -m fx.cli llm probe --model openai/gpt-oss-20b          # second run: cached
PYTHONPATH=. python -m fx.cli llm spend
set -a; source ~/FACET/.env; set +a
PYTHONPATH=. FX_BUDGET=0.05 python -m fx.cli llm call --model deepseek/deepseek-v4-flash --reasoning-off "one word"
```

## Not in this PR

No decomposition, no store tables beyond `call`, no GUI. `pip install -e .` is not required for the checks above; `PYTHONPATH=.` is enough.

## Open questions for review

1. Replies are stored in the `call` table by default (`log_replies=True`) so the cache and the audit trail are one thing. For a large decomposition run that is the bulk of the database; acceptable, or keep replies in a side file?
2. Localhost endpoints are priced at zero unless the model is in the price table. Right for vLLM; wrong if a paid proxy ever runs on localhost.
3. The token estimate for servers that send no usage is characters over 3.8, as in FACET's ledger.
