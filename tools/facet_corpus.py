"""Build data/corpora/facet_prompts.jsonl: every prompt of the FACET harvest, deduplicated by text,
provenance kept as recorded. Pick a domain at import time and a count at run time.

    python tools/facet_corpus.py ~/Documents/FACET/facet_artifact/data/prompt/prompts.jsonl ~/Documents/FACET/runs/batch1/prompts.jsonl
"""
import collections
import json
import sys
from pathlib import Path

KEEP = ("prompt_id", "record_id", "domain", "system_id", "text", "text_sha256", "provenance", "use_case", "role", "stage", "subtask", "family", "collection", "bank_source", "is_prompt")

seen, out = set(), []
for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("is_prompt") is False or not (r.get("text") or "").strip() or r["text_sha256"] in seen:
            continue
        seen.add(r["text_sha256"])
        out.append({k: r[k] for k in KEEP if k in r})
out.sort(key=lambda r: (r["domain"], r["prompt_id"]))
dst = Path(__file__).resolve().parent.parent / "data" / "corpora" / "facet_prompts.jsonl"
with open(dst, "w", encoding="utf-8") as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(out)} prompts, {sum(len(r['text']) for r in out):,} chars, {dict(collections.Counter(r['domain'] for r in out))} -> {dst}")
