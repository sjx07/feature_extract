# feature_extract

Feature extraction from prompt corpora: decomposition of prompts into atomic instructions,
a feature library built from their readings, alignment of new corpora into that library,
and a GUI over the whole thing. Every stage reads and writes one SQLite store; the GUI is
a live view of it.

Stages, in the order the data flows:

| stage | command | writes |
|---|---|---|
| 0 | `fx llm ...` | `call`: every model call with tokens, cost, latency, and a cache |
| 1 | `fx decompose` | `span` (the tree: sections, atoms, material, declined gaps), `reading` (an atom's facets), `decomp` (the run per prompt) |
| 2 | `fx coldstart`, `fx assign` | features, assignments, the leftover rounds, coherence |
| 3 | `fx align` | folds into the seed library, decisions |
| 4 | `fx refine` | splits, edges, reversions, as rules with parameters |
| 5 | `fx serve` | the GUI |

Stages 0 and 1 are in place; the others follow in that order.

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

## Workspace layout

Everything a project produces lives in one workspace directory, the same for the CLI and the
site: `--workspace runs/<name>` or `FX_WORKSPACE`, default `runs/dev`.

```
runs/<name>/
  store.db            the store
  logs/serve.log      the site's log, rotated
  logs/job-<id>.log   one log per run, from the terminal or the site: every prompt finished, every error, the traceback if it died
  uploads/<corpus>/   files dropped into the site or imported from the terminal, kept verbatim
  exports/            anything written out for use elsewhere
```

A run is a `job` row plus its log wherever it was started, so a terminal run shows on the site's
job page and a site run has the same log file. `runs/` is not committed; `data/corpora/` holds
the checked-in corpora: the FACET harvest as one file per domain, 2,591 prompts across six, and two plain-text files.

## Stage 1: decomposition and the site

```
set -a; source ~/FACET/.env; set +a                       # OPENROUTER_API_KEY, OPENAI_API_KEY
PYTHONPATH=. python -m fx.cli -w runs/dev serve --port 8780
```

Open http://localhost:8780. Drop one of the domain files under `data/corpora/facet/`, any FACET
prompts.jsonl (with a domain filter), a folder or zip of text files, or paste one prompt; from a
machine without the files, type a path on the server instead (relative to the repo, `~` allowed). The run panel previews calls, tokens, dollars and, once the store
has timing for the model, time; "run first N" decomposes a pilot, "run all" the rest. The job page
follows the run over server-sent events; the prompt page shows the raw text painted with atoms,
material and declined gaps beside the tree; the queues page lists what to read.

The model is a free-text field on the run panel with the registry's models suggested, and
`--model` on the CLI: `gpt-*` goes to OpenAI, `vendor/model` to OpenRouter, anything else to the
local server. `FX_MODEL` changes the default (`deepseek/deepseek-v4-flash-0731` out of the box),
`FX_MODELS` names a JSON file adding models with their endpoint and price, `--base-url` points a
CLI run at any other server. `fx llm models` prints the list the site shows.

The same from the terminal:

```
PYTHONPATH=. python -m fx.cli -w runs/dev import data/corpora/facet/text2sql.jsonl --name text2sql
PYTHONPATH=. python -m fx.cli -w runs/dev preview --corpus text2sql --limit 30
PYTHONPATH=. python -m fx.cli -w runs/dev decompose --corpus text2sql --limit 30
```

How it decomposes: the REFINE prompt from FACET, applied to the whole prompt and then to every
section, until every leaf is an atom with its facets or material the model marked as such.
Every unowned stretch of a sentence or more is refined once as a gap and its outcome recorded.
Coverage is atom characters over instruction characters, instruction being what the model did
not call material. Reasoning is off by default (the registry knows how to say that to each
endpoint) and replies are constrained to the components schema where the server supports it.

## Tests

```
python -m pytest -q
```

The tests run a scripted OpenAI-compatible server in a thread, so they need no network,
no keys, and no model: replies, empty replies, parameter rejections, denials, streaming,
the cache, the budget, and the pool are all exercised against it.
