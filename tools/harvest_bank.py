"""Build data/corpora/<domain>.jsonl from a prompt-harvest bank.json (the ~/.claude/skills/prompt-harvest format):
one FACET-format record per distinct prompt, provenance and use case kept, the bank's own cleaned text used
when it has one (LaTeX font commands unwrapped, line breaks restored), deduplicated by cleaned text.

    python tools/harvest_bank.py /data/users/jsu323/harvest_entity_resolution/bank.json
    python tools/harvest_bank.py bank.json --domain entity-resolution --out data/corpora/entity-resolution.jsonl
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "corpora"


def rows(bank: dict, domain: str):
    """FACET-format records from a bank, deduplicated by cleaned text (first record wins, later ids kept in meta)."""
    by_text: dict[str, dict] = {}
    for it in bank["prompts"]:
        p = it["prompt"]
        text = (p.get("text_clean") or p.get("text") or "").strip("\n")
        if not text.strip():
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in by_text:
            by_text[h]["duplicate_ids"].append(it["id"])
            continue
        prov = it.get("provenance") or {}
        labels = it.get("labels") or {}
        uc = it.get("use_case") or {}
        by_text[h] = {
            "prompt_id": f"{bank.get('domain', domain)}:{it['id']}", "record_id": it["id"], "domain": domain,
            "system_id": prov.get("source_id"), "text": text, "text_sha256": h,
            "provenance": {k: prov.get(k) for k in ("source_kind", "source_role", "source_id", "url", "commit", "file_path", "line_start", "line_end") if prov.get(k) is not None},
            "use_case": {k: [x.get("value") for x in uc.get(k, [])] for k in ("benchmarks", "models", "harnesses") if uc.get(k)},
            "role": labels.get("stage"), "subtask": labels.get("domain_minor") or labels.get("domain"), "family": labels.get("functionality"),
            "collection": "harvest_" + domain.replace("-", "_"), "bank_source": bank.get("domain"), "is_prompt": True,
            "kind": p.get("kind"), "cleaned_by_harvest": p.get("text_clean_transforms") or [], "duplicate_ids": [],
        }
    return list(by_text.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bank", type=Path)
    ap.add_argument("--domain", default=None, help="domain label (default: the bank's)")
    ap.add_argument("--out", type=Path, default=None, help=f"output file (default {OUT}/<domain>.jsonl)")
    a = ap.parse_args()
    bank = json.load(open(a.bank))
    domain = a.domain or bank.get("domain") or a.bank.parent.name
    out = a.out or OUT / f"{domain}.jsonl"
    rs = rows(bank, domain)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_in = len(bank["prompts"]); dups = sum(len(r["duplicate_ids"]) for r in rs)
    print(f"{n_in} bank prompts -> {len(rs)} distinct ({dups} duplicates folded), {sum(len(r['text']) for r in rs):,} chars, "
          f"{dict(collections.Counter(r['subtask'] for r in rs))} -> {out}")


if __name__ == "__main__":
    main()
