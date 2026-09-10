# Corpora

- `facet_prompts.jsonl`: the whole FACET harvest, 2,591 prompts across six domains after
  deduplication by text (math 1,001, text2sql 806, table-qa 317, science-quantitative 180,
  text2cypher 145, code-generation 142), in FACET format with provenance, system, and use case as
  recorded. Pick a domain at import time (`--domain`, or the domain filter on the drop zone) and a
  count at run time (`--limit`, or "first N only" on the site). Rebuild with
  `python tools/facet_corpus.py <prompts.jsonl ...>`.
- `plain/`: the two harvested copies of the same "query planning optimizer" system prompt as bare
  text files, for the folder importer and for checking that identical text decomposes identically.

```
PYTHONPATH=. python -m fx.cli -w runs/dev import data/corpora/facet_prompts.jsonl --name text2sql --domain text2sql
PYTHONPATH=. python -m fx.cli -w runs/dev decompose --corpus text2sql --limit 30
```
