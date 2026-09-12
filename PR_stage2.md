# Stage 2: the feature library

Branch `stage2-library` against `main`. 55 tests.

## What it does

From a corpus's readings (stage 1) to a codebook of features, one library per kind: guidance readings and
material readings never compete for one feature. The codebook is a tree that only grows; each step is a
command, a button on the site, and a job with a log.

- **collapse**, no calls: identical declarations (polarity + normalised wording) become one *realization*; every
  reading points at its realization, which keeps the prompt quote it came from and the domain nouns the
  decomposition replaced. **embed**, no calls: one vector per realization from a local model (bge-large by default).
- **cold start**, one call per kind over every realization (blocks by verb, a context budget with the tail waiting):
  the codebook as a tree, groups with an aspect and features under them, each with a testable one-sentence
  definition, its polarity and three example wordings, its *anchors*. Anchors are fixed at a node's birth.
- **assign**: new wordings against the whole codebook in batches of 20 (each feature shown with its anchors, each
  wording with its quote), then the open ones against the few nodes nearest by retrieval. Written per batch,
  resumable. After it, the **anchor agreement**: the share of the nodes' own anchors that landed back on them.
- **judge**, read-only: per node, members that do not fit and a split; per group, sibling pairs the members cannot
  tell apart. Flags only; nothing moves on a flag.
- **cluster**, no calls: the open wordings that neighbour each other in the corpus (cosine at a threshold measured
  from the anchors) form candidates when three or more from two or more prompts agree; an open wording with no
  neighbour is marked *specific* and waits, reversibly, for a batch of prompts that brings it one.
- **name**: one call per candidate: a *variant* under a feature (the feature narrowed by a constraint), a *feature*
  under an existing or new group, or a rejection. The members the model kept are assigned to the new node.
- **round**: collapse, embed, cold start if none, assign, judge, then cluster → name → assign the open → judge,
  until no candidate is left. Every step is one line in the job log; a re-run resumes.

Why this shape: the first design revised the whole codebook each round and re-assigned everything. Measured on
entity resolution it drifted (anchor agreement 0.98 → 0.73 → 0.82 across versions) and grew by sharpening
definitions as much as by finding features. Here definitions never change, only the open wordings are ever looked
at again, and "specific" is decided by the corpus (does anything else say it) rather than guessed by a batch.

## Store

`realization` (+ `vector`), `codebook`, `feature` (levels group, feature, variant; `round` says when a node was
added), `assignment` (`note`: specific or named), `flag`; a `realization` column on `reading`.

## Site

Library page per corpus and kind: the summary row (groups, features, variants, rounds, assigned, open, specific,
reading coverage, anchors, flags), the tree with variants under their features and support rolled up, the open
wordings with their status, and the run panel with a cost preview per step. Feature page: definition, group,
support, anchors held or not, flags, members by support, and the readings in their prompts.

## How to check

```
cd ~/Documents/feature_extract && git checkout stage2-library
python -m pytest -q
export HF_HOME=/data/users/jsu323/.cache/huggingface
set -a; source ~/FACET/.env; set +a
PYTHONPATH=. python -m fx.cli -w runs/full serve --port 8780
```

`runs/full` holds the entity-resolution and text2cypher codebooks from the earlier design (versions v1 to v4).
The growing loop starts from the latest version as its root: on the Library page pick entity-resolution, step
"round loop", batch model `deepseek/deepseek-v4-flash-0731`, codebook model `gpt-5.6-sol`, and run. The first
run embeds 2,951 wordings (the model downloads once, about 1.3 GB, into HF_HOME) and then clusters the 1,069
open wordings.

## Open

- The clustering threshold is the 25th percentile of anchor-pair cosines, clamped to [0.6, 0.92]; it needs one
  read on a real corpus.
- A variant is judged like a feature; siblings are only compared within a group, so two variants of one feature
  are not compared with each other yet.
- The embedding runs on CPU unless a GPU is free; 3,000 wordings take about a minute on CPU.
