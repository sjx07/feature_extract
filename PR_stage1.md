# Stage 1: decomposition, the store tables, and the localhost site

Branch `stage1-decompose` against `main`. 30 tests.

## What it does

Drop a corpus into the site, preview the cost, run the decomposition, watch the progress bar,
open any prompt and see its tree beside the raw text, and read the queues of what to check.

- `fx/decompose/prompts.py`: FACET's REFINE prompt v6.6 with one change: material is a component the model emits, with a kind (example, schema, code, slot, template, title, reference, other), rather than text it leaves between components. Plus a system framing that the SPAN is a document to analyze, not a request, and a JSON schema for the reply.
- `fx/decompose/contract.py`, `locate.py`: the tree and the exact-quote locator, ported; case-insensitive fallback when the model re-cases a quote.
- `fx/decompose/profile.py`: recursive refinement, ported, with three changes. Every unowned stretch of a sentence or more is refined once as a gap, wherever it sits, and its outcome is recorded (FACET's marker-and-length heuristic is gone). A re-ask can only replace the first reply when it parses and locates at least as many components, so a refusal or broken JSON on the re-ask never erases a usable reply. An unparsable root leaves one unrefined leaf so the prompt shows in the queue. The model alone decides what is material; nothing overrides it.
- `fx/decompose/runner.py`: `preview` (calls, tokens, dollars from the registry, time once the store has timing for the model) and `run` (thread pool over prompts, per-endpoint admission equal to the worker count, resumable, one transaction per prompt, `progress` callback, `Stop`).
- `fx/corpus.py`: importers for a FACET prompts.jsonl (with a domain filter), a folder or zip of text files, one text file, or pasted text; idempotent by text hash.
- `fx/store.py`: every table in one schema: corpus, prompt, span, reading, decomp, job. `span` holds every located node of a tree (sections, atoms, material, unrefined leaves, and the gaps the model declined, with the material kind or gap outcome in `note`); `reading` holds an atom's facets, the unit stage 2 assigns; `decomp` is the stage's record per prompt with no counts that are one query away.
- `fx/llm`: `schema=` on `complete` sends `response_format` json_schema, dropped per model if the server rejects it; `reasoning_off(model)` in the registry maps to what each endpoint honours (measured: vLLM's gpt-oss takes a top-level `reasoning_effort` and ignores OpenRouter's `reasoning.enabled`).
- `fx/gui`: FastAPI over the store. Corpora (drop zone, run panel with preview), job page (progress over server-sent events, projected remaining time, spend, stop), prompts, prompt page (raw text painted with atoms, material and declined gaps; the tree beside it, sections collapsible, hover links both ways), queues (low coverage, declined gaps, unrefined leaves, failures).
- `fx/unwrap.py`: harvest residue. A prompt that arrived inside the code or record it was lifted from (an argparse default, a string assignment, an f-string with slots, a docstring, a Lean literal, a JSON field, a one-line string with escaped newlines) is unwrapped at import, exactly and without a model call; the original is kept in the prompt's meta, the prompt page shows it, the queues page lists the unwrapped prompts. 116 of the 2,591 corpus prompts; dedup runs on the unwrapped text so a wrapped and a clean copy are one prompt.
- `fx/util/jsonx.py`: reply parsing shared by every stage.
- `fx/paths.py`, `fx/jobs.py`: the workspace layout (store, logs, uploads, exports under `runs/<name>`) and one job lifecycle for the CLI and the site: a `job` row, a per-job log with every prompt and any traceback, progress into both.
- `data/corpora/`: the whole FACET harvest checked in as one file per domain, 2,591 prompts across six with provenance, and two plain-text files; drop a domain file in and choose the count at run time. `tools/facet_corpus.py` rebuilds them.

## What the pilot showed (text2sql, 20 prompts, gpt-oss-20b on the local vLLM)

Before the fixes above: two calls at 74 s hitting the reply cap with reasoning on; a refusal on a re-ask overwriting a good reply; a fence rule that forced an output template to material. After: 145 calls at 8.7 s, no truncation, no errors; coverage 0.92 or better on 14 of 20, three under 0.6. The three share one cause, quotes the locator cannot place because the model copies long or repeated stretches instead of three words; that is the next thing to fix and it belongs in the re-ask, not in a rule.

## Removed on review

`rules.py` and then `checks.py`: regex-made atoms for role sentences, forced material for fenced code, and a formulaic-sentence check. The pilot showed 11 of 11 role sentences inside model atoms, and the fence rule did harm; what the model leaves unowned is visible in the queues without predicting its wording.

## How to check

```
cd ~/Documents/feature_extract
python -m pytest -q
set -a; source ~/FACET/.env; set +a
PYTHONPATH=. python -m fx.cli -w runs/dev serve --port 8780
```

Then http://localhost:8780: drop `data/corpora/facet/text2sql.jsonl`, preview, run first 30.

## Reasoning

The model reasons; there is no switch. FACET's v5 library was decomposed with DeepSeek's reasoning on (provider default, 32k ceiling, one low-effort fallback when a call exhausted it): 5.2 calls and 593 s per prompt. Its pilot v9 read reasoning off as worse (content typed as guidance, sentences cut, polarity inverted). The stage 1 comparison on 30 text2sql prompts (`runs/cmp_off`, `runs/cmp_on`) agreed: off needed 2 re-asks per prompt and 76 quote problems and looped to the ceiling on 19 calls; on needed 0.3 re-asks and 9 quote problems, at four times the wall time and cost. In the code: 32k reply ceiling, a reply that ended by `length` with no text is not retried but answered once more at low effort, a leaf the model emitted without facets gets one call for them, and the locator's re-ask asks for a short exact quote.

Stop now works while calls are in flight: the button marks the job `stopping`, prompts not yet started are cancelled, the HTTP clients are closed so waiting calls return, and prompts caught mid-way are left to do for the next run.

## Open

- Quote failures on long prompts: teach the re-ask to name the offending components and ask for three-word quotes, and measure the re-ask rate (1.85 per prompt in the pilot).
- A same-corpus comparison against DeepSeek v4 flash, which decomposed the v5 library, is one preview and one run away.
- The material share needs a sampled read on text2sql, whose prompts are mostly schema and examples by volume.
