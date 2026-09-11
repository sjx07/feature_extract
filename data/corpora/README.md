# Corpora

- `facet/<domain>.jsonl`: the FACET harvest, one file per domain, deduplicated by text across the
  whole harvest, in FACET format with provenance, system, and use case as recorded:

  | file | prompts | chars |
  |---|---|---|
  | math.jsonl | 1,001 | 1,428,192 |
  | text2sql.jsonl | 806 | 1,113,932 |
  | table-qa.jsonl | 317 | 946,193 |
  | science-quantitative.jsonl | 180 | 216,379 |
  | text2cypher.jsonl | 145 | 181,570 |
  | code-generation.jsonl | 142 | 151,614 |

  About 116 of them arrived inside the code or record they were harvested from (a Python
  string, an argparse default, a Lean literal, a JSON field, or a one-line string with escaped
  newlines); the importer unwraps those and keeps the original in the prompt's meta, and the
  queues page lists them.
  Drop one file onto the site or import it from the terminal; pick the count at run time
  (`--limit`, or "first N only" on the site). Rebuild with
  `python tools/facet_corpus.py <prompts.jsonl ...>` (`--domains` to write a subset).
- `entity-resolution.jsonl`: 247 prompts from the prompt-harvest bank at
  `/data/users/jsu323/harvest_entity_resolution/bank.json` (256 records, 9 duplicates folded), in the same
  FACET format with provenance, use case, and the harvest's sub-task label as `subtask`; the bank's own
  cleaned text (LaTeX font commands unwrapped) is used where it has one. Rebuild with
  `python tools/harvest_bank.py <bank.json>`. Held out from the six development domains.
- `plain/`: the two harvested copies of the same "query planning optimizer" system prompt as bare
  text files, for the folder importer and for checking that identical text decomposes identically.

```
PYTHONPATH=. python -m fx.cli -w runs/dev import data/corpora/facet/text2sql.jsonl --name text2sql
PYTHONPATH=. python -m fx.cli -w runs/dev decompose --corpus text2sql --limit 30
```
