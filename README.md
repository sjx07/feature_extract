# feature_extract

Feature extraction from prompt corpora: decomposition of prompts into atomic instructions,
a feature library built from their readings, alignment of new corpora into that library,
and a GUI over the whole thing. Every stage reads and writes one SQLite store; the GUI is
a live view of it.

Stages, in the order the data flows:

| stage | command | writes |
|---|---|---|
| 0 | `fx llm ...` | `call`: every model call with tokens, cost, latency, and a cache |
| 1 | `fx decompose` | prompts, spans, atoms, readings, gaps, coverage |
| 2 | `fx coldstart`, `fx assign` | features, assignments, the leftover rounds, coherence |
| 3 | `fx align` | folds into the seed library, decisions |
| 4 | `fx refine` | splits, edges, reversions, as rules with parameters |
| 5 | `fx serve` | the GUI |

Stage 0 is in place; the others are planned in that order.

## Stage 0: models, ledger, cache

`fx.llm.Client` talks to any OpenAI-compatible endpoint and returns a `Reply` rather than
raising for model-side failures. It probes and remembers hosted parameter shapes, retries
empty replies and transport failures, classifies denials and unreachable endpoints so they
are never logged as paid calls, streams long generations, serves repeated prompts from the
store's cache at no cost, prices every paid call, and refuses to start a call once the
budget is spent. `run_many` runs a list of prompts on a thread pool with per-endpoint
admission and a progress callback; `with_fallback` chains model settings.

```python
from fx.store import Store
from fx.llm import Client, run_many

store = Store("runs/demo/store.db")
client = Client(store, budget=20.0)
r = client.complete("Reply with one word.", model="openai/gpt-oss-20b", stage="probe")
r.text, r.usage, r.cost, r.latency, r.cached, r.error

replies = run_many(client, prompts, model="deepseek/deepseek-v4-pro", workers=8, stage="assign",
                   extra_body={"reasoning": {"enabled": False}}, progress=lambda done, total: print(done, total))
```

Models resolve to endpoints by name: `gpt-*` to OpenAI, `vendor/model` to OpenRouter,
anything else to the local vLLM server at `FX_LOCAL_URL` (default `http://localhost:8000/v1`).
`--base-url` or `Client(base_url=...)` overrides that for any other server. Keys come from
`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `FX_API_KEY`. A JSON file named by `FX_MODELS`
adds models with their own endpoint and price without a code change. Unknown hosted models
are priced at the top rate so a missing entry over-counts.

```
fx llm models                                   where each model resolves and its price
fx llm probe --model openai/gpt-oss-20b         one call against the local server
fx llm call --model gpt-5.6-sol --stream "..."  a priced, cached call
fx llm spend                                    dollars by model from the store
```

## Tests

```
python -m pytest -q
```

The tests run a scripted OpenAI-compatible server in a thread, so they need no network,
no keys, and no model: replies, empty replies, parameter rejections, denials, streaming,
the cache, the budget, and the pool are all exercised against it.
