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
| 2 | `fx library round` (coldstart, assign, judge, cluster, name) | `realization`, `vector`, `codebook`, `feature` (the tree: groups, features, variants), `assignment`, `flag` |
| 3 | `fx align round` (embed, assign, judge, cluster, name) | `alignment`, `fvector`; the seed codebook's global features in `feature` |
| 4 | `fx refine` | splits, edges, reversions, as rules with parameters |
| 5 | `fx serve` | the GUI |

Stages 0 to 3 are in place; the others follow in that order.

## The base library

`library/base/store.db` is a released library: seven FACET corpora (3,027 prompts, decomposed), a codebook for four of
them (math, science-quantitative, text2cypher, text2sql: 1,214 features) and the seed of 122 global features over them;
tags, imports and profiles as the site expects. Its model-call ledger, job rows and wording embeddings are stripped
(the embeddings come back from the local encoder with `fx library embed`, no model call). Serve it:

```
PYTHONPATH=. python -m fx.cli -w library/base serve --port 8780      # then http://localhost:8780
```

A run from the site writes into that workspace; copy the directory first to keep the base untouched.

## Stage 0: models, ledger, cache

`fx.llm.Client` talks to any OpenAI-compatible endpoint and returns a `Reply` rather than
raising for model-side failures. It probes and remembers hosted parameter shapes, retries
empty replies and transport failures, classifies denials and unreachable endpoints so they
are never logged as paid calls, streams long generations, serves repeated prompts from the
store's cache at no cost, prices every paid call, and refuses to start a call once the
budget is spent. `run_many` runs a list of prompts on a thread pool with per-endpoint
admission and a progress callback; `with_fallback` chains model settings.

```python
from fx.core.store import Store
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
has timing for the model, time; "run first N" decomposes a pilot, "run all" the rest. `workers` is the
number of calls in flight across the run (OpenRouter has taken 512 without throttling); within a prompt the
sibling sections, gaps and leaf checks of a node are refined together, four at a time. The job page
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

## Stage 2: the feature library

One library per corpus and kind (guidance, material), a tree that only grows. `collapse` folds identical
declarations into realizations and `embed` gives each one a vector from a local model; `coldstart` writes
the codebook in one whole-corpus call (groups with an aspect, features with a testable definition, polarity
and example wordings, the anchors); `assign` puts every wording on a node or leaves it open, first against
the whole codebook and then, for the open ones, against the few nodes nearest by retrieval; `judge` reports
misfits, splits and indistinct siblings without moving anything. Then the loop: `cluster` gives every open
wording not yet looked at its neighbourhood, itself and its nearest open wordings from other prompts (no
threshold: retrieval orders, the model decides); `name` reads each neighbourhood in parallel and proposes a variant
under a feature, a new feature, or rejects it, and a rejected seed is marked specific; `join`, one call a round,
sees every proposal beside the tree and decides what is new, what is an existing feature under another wording,
and what two proposals said twice, then places and founds groups; assign runs again over the open wordings; until
every open wording has had its look. Nothing written is ever rewritten. The
site's Library page runs each step with a cost preview and shows the tree, the open wordings and the flags.

```
export HF_HOME=/data/users/$USER/.cache/huggingface     # the embedding model downloads here, not into the home quota
PYTHONPATH=. python -m fx.cli -w runs/dev library round  --corpus text2sql --kind guidance --model deepseek/deepseek-v4-flash-0731 --codebook-model gpt-5.6-sol
PYTHONPATH=. python -m fx.cli -w runs/dev library status --corpus text2sql --kind guidance
```

## Stage 3: the seed library

A run from the site (a corpus under a profile) does stages 1 and 2 and stops; stage 3 runs only when the profile's
`stage 3` is `on`, and then only once a second corpus has a codebook of the kind. A new corpus is studied on its own first.

Stages 2 and 3 are one loop, `fx/loop`, run on two levels: the wordings of a corpus (`fx/library`)
and the features of every corpus (`fx/align`). A `Level` carries what differs; the engine carries assign,
judge, reopen, cluster, name and the settle loop.

The stage 2 loop one level up. The units are the per-corpus features of every library of a kind, read as cards
(name, definition, anchors, corpus, support); a global feature is one instruction several corpora give under
their own domain nouns. `embed` vectors the cards; `assign` puts open cards on the nearest globals or none;
`cluster` gives every open card not yet looked at its neighbourhood, itself and its nearest open cards from other
corpora; `name` reads it and proposes a global feature (named without domain nouns, defined across domains,
members from two or more corpora) or rejects it, and a rejected card is domain-specific; `join` reconciles the
round's proposals against the seed; `judge` flags members whose wordings give another instruction,
and reopen sends first-time misfits back with the reason and turns a split into variants under the node. The Seed page shows the globals with their members per
corpus. Variants stay under their features: global → per-corpus feature → variant.

```
PYTHONPATH=. python -m fx.cli -w runs/full align cluster
PYTHONPATH=. python -m fx.cli -w runs/full align round --model deepseek/deepseek-v4-flash-0731 --codebook-model gpt-5.6-sol
PYTHONPATH=. python -m fx.cli -w runs/full align status
```

## Layout, the store's version, history

```
fx/core     store, paths, jobs, settings, migrations, llm/, util/     what every stage and the site share
fx/data     corpus, unwrap, tags, history                             corpora and prompts as data
fx/ingest   decompose/, induce/, generalize/, loop/                   the stages: readings, a corpus's codebook, the seed
fx/views    library, ingest                                           the two pages' queries
fx/gui      server, static/                                           the site
```

A store carries a schema version (`PRAGMA user_version`); opening it runs the numbered steps in `fx/core/migrations.py`
above its version, each in one transaction, and a store ahead of the code refuses to open. `fx store version` prints
both. Since version 6: imports are events (`import`), tags are rows (`tag`, with domain / system / task also cached as
prompt columns), the seed is a codebook with scope `seed` and no corpus, and the legacy tables are gone.

`fx history checkpoint | list | diff | restore | branch`: a checkpoint is one corpus's rows (or the seed's line) as
content-addressed blobs under `runs/history/objects`; a run from the site checkpoints its corpus and the seed first.

## Tests

```
python -m pytest -q
```

The tests run a scripted OpenAI-compatible server in a thread, so they need no network,
no keys, and no model: replies, empty replies, parameter rejections, denials, streaming,
the cache, the budget, and the pool are all exercised against it.
