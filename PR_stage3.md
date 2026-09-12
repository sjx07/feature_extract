# Stage 3: the seed library (alignment across corpora)

Branch `stage3-align` against `main`. 59 tests (3 new, in `tests/test_align.py`, on two seeded libraries and the scripted server).

## One loop, two levels

Stage 2 and stage 3 ran the same algorithm on two unit types, so the loop is now written once, in `fx/loop`,
against a `Level` (`fx/loop/level.py`) that says what differs: the units (a corpus's wordings, or every corpus's
features as cards), how they group for support (prompts, or corpora), how a node's vector is formed (its anchors, or
its members), the prompt texts, and the naming minimums. `fx/library` keeps collapse and the cold start and defines the
wording level; `fx/align` keeps the cards and the seed codebook and defines the feature level; both keep their old
function signatures as thin wrappers. One `membership` table (unit kind, unit, codebook → node) and one `embedding`
table replace `assignment`/`alignment` and `vector`/`fvector`, which are copied into them on open and left in place.
The thirteen step modules of the two packages (1,057 lines) became the engine and two level files (665 lines); the
59 tests are unchanged in what they check.

## What it does

The stage 2 loop one level up. The units are the per-corpus features of every library of a kind, each read as a
card: name, definition, polarity, anchors, corpus, support. A global feature is one instruction that several
corpora give under their own domain nouns; the seed library is a codebook on the corpus named `seed` whose feature
rows are the globals, and `alignment` maps each per-corpus feature to one global or none. Variants are not aligned:
they stay under their feature, so the hierarchy is global → per-corpus feature → variant, and support rolls up.

- **embed**: one vector per card (the same local model as the wordings; table `fvector`).
- **assign**: open cards against the few globals nearest by retrieval (a global's vector is the mean of its members'),
  in batches sharing a nearest global; one global or null per card. A card the judge removed from a global carries
  the reason and that global is offered back; putting it back settles the flag.
- **cluster**, no calls: every open card not yet looked at is a seed; its candidate is itself and its five nearest open
  cards from *other* corpora with the same polarity (same-corpus neighbours are not cross-corpus support). No threshold:
  a seed of 609 cards born at cosine 0.84 gave 23 globals and then nothing, while two of them absorbed 96 cards. A
  seed the namer does not place is *domain-specific*, reversibly: it stays in the pool as a neighbour for later seeds.
- **name**: one call per candidate: the same global feature, named without domain nouns and defined across domains,
  under an existing or new seed group, with the member ids it accepts (from at least two corpora), or a rejection.
- **judge**, read-only: per global, the members with a sample of the wordings each covers; members whose wordings
  give another instruction are flagged (feature = the global, other = the member). Flags repeated after being acted
  on are standing.
- **round**: embed, assign, judge, then reopen → cluster → name → assign → judge until settled (no new flag, no
  candidate) or the round limit.

Nothing in a corpus library changes. The FACET v5 library is not the root: it enters as one more library to align
when imported, which keeps the held-out comparison (how many v5 features find a counterpart, and which new features
v5 never had).

## Store

`membership` with kind `feature` (per-corpus feature → global, note: specific / named / reopened:<id>|why), `embedding`
with kind `feature`; the seed codebook and its globals in `codebook` and `feature`; flags on globals in `flag` with `other` = the member.

## Site

Seed page per kind: the summary (globals, aligned cards, domain-specific, open, flags), a per-corpus table (features,
aligned share, prompts under aligned features), the globals as a tree with members per corpus and their flags, the
open and domain-specific cards, and the run panel. A feature page says which global it is aligned to.

## How to check

```
cd ~/Documents/feature_extract && git checkout stage3-align
python -m pytest -q
export HF_HOME=/data/users/jsu323/.cache/huggingface CUDA_VISIBLE_DEVICES=1
set -a; source ~/FACET/.env; set +a
PYTHONPATH=. python -m fx.cli -w runs/full align cluster          # no calls: the candidates across text2sql, math, entity-resolution, text2cypher
PYTHONPATH=. python -m fx.cli -w runs/full align round --model deepseek/deepseek-v4-flash-0731 --codebook-model gpt-5.6-sol --rounds 5 --budget 5
```

`runs/full` holds four guidance libraries (text2sql 224 features, math 225, entity-resolution 87, text2cypher 73):
about 600 cards, so a round is a few hundred short calls; naming is the cost, about 2 cents a cluster.

## Open

- A narrower feature (a global's instruction plus a rule) is not an instance and becomes a global of its own; the
  narrower-than relation between globals is not recorded yet.
- Granularity mismatches (a variant in one corpus, a feature in another) surface as domain-specific features; a
  variant-to-feature pass is the fix.
- Importing the FACET v5 library as a corpus library (features with definitions and example wordings) is the step
  that makes the held-out comparison.
