# Stage 2: the feature library

Branch `stage2-library` against `main`. 44 tests (4 new files' worth: collapse, the four steps, the site).

## What it does

From a corpus's readings (stage 1) to a codebook of features, one library per kind: guidance readings and
material readings never compete for one feature. Five steps, each a command and a button, each a job with a log:

- **collapse**, no calls: identical declarations (polarity + normalised wording) become one *realization*; every
  reading points at its realization. Assignment works on realizations, so a wording that recurs is judged once.
- **cold start**, one call per kind over every realization of the corpus: the codebook as a tree, groups with an
  aspect and features under them, each feature with a testable one-sentence definition, its polarity and three
  example realizations. Whole-corpus context by design: batch induction is what inflated v5 before cover merged it.
- **assign**, batches of 40 realizations against the codebook, feature or none with a confidence. Resumable, runs
  from scratch on each version, so no assignment depends on the path that produced it. After it, the version's
  **anchor agreement** is measured: the share of features' own examples that landed back on them.
- **judge**, read-only: per feature, members that do not fit and a split when the members fall apart; per group,
  sibling pairs the members cannot tell apart. Flags only; nothing moves on a flag.
- **revise**, the one writer after the cold start: the codebook, the leftovers (none or low confidence) and the
  flags in one call, returning the next version. A feature whose meaning is unchanged keeps its id (`prev` links
  the versions); a changed or merged one is new and names what it replaces. Then assign again on the new version.

The hierarchy is in the data: `feature` rows at level group and level feature, parent from feature to group,
aspect on the group from a closed list; nothing is assigned to a group, its support is the union of its features.

## Store

`realization`, `codebook` (the version record, with the anchor agreement), `feature` (the tree, versioned),
`assignment`, `flag`; plus a `realization` column on `reading`, added by a migration on open.

## Site

Library page per corpus and kind: the versions table (groups, features, assigned, leftover, low-confidence,
reading coverage, anchors, flags), the codebook as a collapsible tree with support and flags per feature, the
leftover list, and the run panel with a cost preview per step. Feature page: definition, group, support, whether
its anchors held, lineage across versions, flags, members by support, and the readings in their prompts.

## How to check

```
cd ~/Documents/feature_extract && git checkout stage2-library
python -m pytest -q
set -a; source ~/FACET/.env; set +a
PYTHONPATH=. python -m fx.cli -w runs/pilot_s2 serve --port 8781
```

`runs/pilot_s2` is a copy of `runs/v2` (128 text2sql prompts decomposed) with the pilot library on it: open
Library, read the v1 and v2 codebooks and the flags. From the terminal the same steps are
`fx library coldstart|assign|judge|revise --corpus text2sql --kind guidance|material`.

## Open

- Cold-start model: `gpt-5.6-sol` by default; the pilot says whether it needs the 1M context or DeepSeek v4 pro does as well.
- Stop for the library steps cancels between batches; a cold start or revise call cannot be interrupted.
- The site's feature page lists readings; painting them into the prompt view is a link away, not inline yet.
