"""The granularity read of the seed, for a person: globals by size, their members per corpus, the concentration report,
the join's verdicts, the judge's pairs. Reads only.

    PYTHONPATH=. python tools/read_seed.py runs/fresh [guidance]
"""
import json
import random
import sqlite3
import sys
from pathlib import Path

ws = sys.argv[1]; kind = sys.argv[2] if len(sys.argv) > 2 else "guidance"
c = sqlite3.connect(f"file:{Path(ws) / 'store.db'}?mode=ro", uri=True); c.row_factory = sqlite3.Row
seed = c.execute("SELECT id FROM corpus WHERE name='seed'").fetchone()[0]
cb = c.execute("SELECT id FROM codebook WHERE corpus=? AND kind=?", (seed, kind)).fetchone()[0]
corpus_name = {r["id"]: r["name"] for r in c.execute("SELECT id, name FROM corpus")}
cards = {r["id"]: dict(r) for r in c.execute("SELECT f.id, f.name, f.definition, cb.corpus FROM feature f JOIN codebook cb ON cb.id=f.codebook WHERE f.level='feature' AND cb.corpus != ? AND cb.kind=? AND cb.id IN (SELECT MAX(id) FROM codebook GROUP BY corpus, kind)", (seed, kind))}
mem = {}
for r in c.execute("SELECT unit, node, note FROM membership WHERE kind='feature' AND codebook=?", (cb,)):
    mem.setdefault(r["node"], []).append(int(r["unit"]))
per_corpus = {}
for k, card in cards.items():
    per_corpus[card["corpus"]] = per_corpus.get(card["corpus"], 0) + 1
globals_ = [dict(r) for r in c.execute("SELECT f.id, f.name, f.definition, f.polarity, f.round, g.name gname FROM feature f JOIN feature g ON g.id=f.parent WHERE f.codebook=? AND f.level='feature'", (cb,))]
print(f"== seed {kind} (codebook {cb}): {len(globals_)} globals; cards {len(cards)}; aligned {sum(len(v) for k, v in mem.items() if k is not None)}; open {len(mem.get(None, []))}")
print("by round:", [dict(r) for r in c.execute("SELECT round, SUM(level='feature') f, SUM(level='retired') retired FROM feature WHERE codebook=? GROUP BY round ORDER BY round", (cb,))])
for g in globals_:
    ms = [cards[u] for u in mem.get(g["id"], []) if u in cards]
    g["members"] = ms; g["corpora"] = len({m["corpus"] for m in ms})
    g["share"] = max(((sum(1 for m in ms if m["corpus"] == cid) / per_corpus[cid], cid) for cid in {m["corpus"] for m in ms}), default=(0, None))
print("\nlargest 12 globals (members / corpora / largest share of one corpus's features):")
for g in sorted(globals_, key=lambda g: -len(g["members"]))[:12]:
    print(f"   {len(g['members']):>3}m {g['corpora']}c share {g['share'][0]:.2f} of {corpus_name.get(g['share'][1], '?')[:12]:12} r{g['round']} ({g['polarity'][:3]}) {g['name'][:60]}")
    for m in sorted(g["members"], key=lambda m: m["corpus"])[:6]:
        print(f"         [{corpus_name.get(m['corpus'], '?')[:12]}] {m['name'][:70]}")
random.seed(7)
print("\n10 globals at random:")
for g in random.sample(globals_, min(10, len(globals_))):
    print(f"   {len(g['members']):>3}m {g['corpora']}c r{g['round']} {g['name'][:60]:60} | {g['gname']}")
    for m in sorted(g["members"], key=lambda m: m["corpus"])[:4]:
        print(f"         [{corpus_name.get(m['corpus'], '?')[:12]}] {m['name'][:70]}")
print("\nflags:", [dict(r) for r in c.execute("SELECT verdict, standing, COUNT(*) k FROM flag WHERE codebook=? GROUP BY 1,2", (cb,))])
for r in c.execute("SELECT a.name an, b.name bn, f.standing FROM flag f JOIN feature a ON a.id=f.feature JOIN feature b ON b.id=f.other WHERE f.codebook=? AND f.verdict='indistinct' LIMIT 10", (cb,)):
    print(f"   indistinct [{'standing' if r['standing'] else 'new'}] {r['an'][:40]:40} ~ {r['bn'][:40]}")
print("\ncost of the seed's calls:", c.execute("SELECT ROUND(SUM(COALESCE(billed,cost,0)),2) FROM call WHERE note LIKE ? AND at >= (SELECT at FROM codebook WHERE id=?)", (f"seed:{kind}:%", cb)).fetchone()[0])
