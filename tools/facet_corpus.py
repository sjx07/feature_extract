"""Build data/corpora/facet/<domain>.jsonl: every prompt of the FACET harvest, one file per domain,
deduplicated by text across the whole harvest, provenance kept as recorded. Drop one file into the
site or import it from the terminal; pick the count at run time.

    python tools/facet_corpus.py ~/Documents/FACET/facet_artifact/data/prompt/prompts.jsonl ~/Documents/FACET/runs/batch1/prompts.jsonl
    python tools/facet_corpus.py --domains text2sql math -- <prompts.jsonl ...>      # only these domains
"""
import argparse
import collections
import json
from pathlib import Path

KEEP = ("prompt_id", "record_id", "domain", "system_id", "text", "text_sha256", "provenance", "use_case", "role", "stage", "subtask", "family", "collection", "bank_source", "is_prompt")
OUT = Path(__file__).resolve().parent.parent / "data" / "corpora" / "facet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="FACET prompts.jsonl files, earlier files win on duplicate text")
    ap.add_argument("--domains", nargs="*", default=None, help="domains to write (default: every domain found)")
    ap.add_argument("--out", type=Path, default=OUT, help=f"output directory (default {OUT})")
    args = ap.parse_args()

    seen, by_domain = set(), collections.defaultdict(list)
    for path in args.paths:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("is_prompt") is False or not (r.get("text") or "").strip() or r["text_sha256"] in seen:
                continue
            if args.domains and r.get("domain") not in args.domains:
                continue
            seen.add(r["text_sha256"])
            by_domain[r["domain"]].append({k: r[k] for k in KEEP if k in r})

    args.out.mkdir(parents=True, exist_ok=True)
    for domain in sorted(by_domain):
        rows = sorted(by_domain[domain], key=lambda r: r["prompt_id"])
        dst = args.out / f"{domain}.jsonl"
        with open(dst, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{domain:24s} {len(rows):5d} prompts {sum(len(r['text']) for r in rows):12,} chars -> {dst}")
    print(f"{sum(map(len, by_domain.values()))} prompts in {len(by_domain)} domains")


if __name__ == "__main__":
    main()
