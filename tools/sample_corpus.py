"""Build data/corpora/facet_sample_30.jsonl: five prompts per FACET domain, chosen deterministically
(the median-length prompt of each of five length quantiles per domain), provenance kept as recorded.

    python tools/sample_corpus.py ~/Documents/FACET/facet_artifact/data/prompt/prompts.jsonl ~/Documents/FACET/runs/batch1/prompts.jsonl
"""
import json
import sys
from pathlib import Path

PER_DOMAIN = 5
KEEP = ("prompt_id", "record_id", "domain", "system_id", "text", "text_sha256", "provenance", "use_case", "role", "stage", "subtask", "family", "collection", "bank_source", "is_prompt")

rows = []
for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("is_prompt") is not False and (r.get("text") or "").strip():
                rows.append(r)
seen, uniq = set(), []
for r in rows:
    if r["text_sha256"] not in seen:
        seen.add(r["text_sha256"]); uniq.append(r)
out = []
for dom in sorted({r["domain"] for r in uniq}):
    rs = sorted([r for r in uniq if r["domain"] == dom], key=lambda r: (len(r["text"]), r["prompt_id"]))
    n = len(rs)
    for q in range(PER_DOMAIN):                      # the middle of each of five length bands
        lo, hi = q * n // PER_DOMAIN, (q + 1) * n // PER_DOMAIN
        out.append(rs[(lo + hi) // 2])
dst = Path(__file__).resolve().parent.parent / "data" / "corpora" / "facet_sample_30.jsonl"
with open(dst, "w", encoding="utf-8") as fh:
    for r in out:
        fh.write(json.dumps({k: r[k] for k in KEEP if k in r}, ensure_ascii=False) + "\n")
print(f"{len(out)} prompts, {sum(len(r['text']) for r in out):,} chars -> {dst}")
