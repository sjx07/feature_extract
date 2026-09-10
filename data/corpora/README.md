# Test corpora

Small corpora checked in for tests, demos, and a first run on a fresh machine. Nothing here is
a substitute for a real corpus; the FACET artifact's prompts.jsonl files stay where they are.

- `facet_sample_30.jsonl`: thirty prompts from the FACET harvest, five per domain (code-generation,
  math, science-quantitative, table-qa, text2cypher, text2sql), the median prompt of each of five
  length bands per domain, so lengths run from about 140 to 8,400 characters. FACET format, with
  provenance, system, and use case as recorded. Rebuild with `python tools/sample_corpus.py <prompts.jsonl ...>`.
- `plain/`: two prompts as bare text files, the two harvested copies of the same "query planning
  optimizer" system prompt, for the folder importer and for checking that identical text
  decomposes identically.

```
PYTHONPATH=. python -m fx.cli import data/corpora/facet_sample_30.jsonl --name sample
PYTHONPATH=. python -m fx.cli preview --corpus sample
```
