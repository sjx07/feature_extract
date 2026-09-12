"""The granularity read of one library, for a person: what each round added, how big the nodes are, what the judge and
the join think, the largest and a random sample of the loop-born features. Reads only.

    PYTHONPATH=. python tools/read_library.py runs/fresh text2sql [guidance]
"""
import collections
import random
import sqlite3
import sys
from pathlib import Path

ws, corpus = sys.argv[1], sys.argv[2]
kind = sys.argv[3] if len(sys.argv) > 3 else "guidance"
c = sqlite3.connect(f"file:{Path(ws) / 'store.db'}?mode=ro", uri=True); c.row_factory = sqlite3.Row
cid = c.execute("SELECT id FROM corpus WHERE name=?", (corpus,)).fetchone()[0]
cb = c.execute("SELECT id, version FROM codebook WHERE corpus=? AND kind=? ORDER BY version DESC", (cid, kind)).fetchone()
cbid = cb["id"]
print(f"== {corpus} {kind} v{cb['version']} (codebook {cbid})")
n_real = c.execute("SELECT COUNT(*) k, COALESCE(SUM(n),0) n FROM realization WHERE corpus=? AND kind=?", (cid, kind)).fetchone()
m = c.execute("SELECT SUM(node IS NOT NULL) placed, SUM(node IS NULL) open, SUM(node IS NULL AND note='specific') specific FROM membership WHERE kind='realization' AND codebook=?", (cbid,)).fetchone()
cov = c.execute("SELECT ROUND(1.0*SUM(CASE WHEN m.node IS NOT NULL THEN r.n ELSE 0 END)/SUM(r.n),3) FROM realization r LEFT JOIN membership m ON m.unit=r.id AND m.kind='realization' AND m.codebook=? WHERE r.corpus=? AND r.kind=?", (cbid, cid, kind)).fetchone()[0]
print(f"wordings {n_real['k']} ({n_real['n']} readings); placed {m['placed']}, open {m['open']} of which specific {m['specific']}; reading coverage {cov}")
print("by round:", [dict(r) for r in c.execute("SELECT round, SUM(level='feature') f, SUM(level='variant') v, SUM(level='group') g, SUM(level='retired') retired FROM feature WHERE codebook=? GROUP BY round ORDER BY round", (cbid,))])
rows = c.execute("""SELECT f.id, f.name, f.level, f.round, f.parent, p.name pname, p.level plevel,
      (SELECT COUNT(*) FROM membership m WHERE m.kind='realization' AND m.codebook=? AND m.node=f.id) w,
      (SELECT COALESCE(SUM(r.prompts),0) FROM membership m JOIN realization r ON r.id=m.unit WHERE m.kind='realization' AND m.codebook=? AND m.node=f.id) p
      FROM feature f LEFT JOIN feature p ON p.id=f.parent WHERE f.codebook=? AND f.level IN ('feature','variant')""", (cbid, cbid, cbid)).fetchall()
feats = [r for r in rows if r["level"] == "feature"]; vars_ = [r for r in rows if r["level"] == "variant"]
print(f"nodes: {len(feats)} features, {len(vars_)} variants; unplaced features: {sum(1 for r in feats if r['parent'] is None)}")
print("wordings per feature:", dict(sorted(collections.Counter(min(r["w"], 20) // 5 * 5 for r in feats).items())), "(bucketed by 5, 20+ together)")
print("features with fewer than 3 wordings:", sum(1 for r in feats if r["w"] < 3), " with 0:", sum(1 for r in feats if r["w"] == 0))
print("\nlargest 12 features:")
for r in sorted(feats, key=lambda r: -r["p"])[:12]:
    print(f"   {r['w']:>3}w {r['p']:>4}p r{r['round']} {r['name'][:64]:64} | {str(r['pname'] or 'unplaced')[:26]}")
loop = [r for r in feats if r["round"] and r["round"] > 0]
random.seed(7)
print(f"\n20 loop-born features at random (of {len(loop)}):")
for r in random.sample(loop, min(20, len(loop))):
    print(f"   {r['w']:>3}w {r['p']:>4}p r{r['round']} {r['name'][:64]:64} | {str(r['pname'] or 'unplaced')[:26]}")
print("\nvariants, 10 at random:")
for r in random.sample(vars_, min(10, len(vars_))):
    print(f"   {r['w']:>3}w r{r['round']} {r['name'][:50]:50} under {str(r['pname'])[:40]}")
dups = [tuple(r) for r in c.execute("SELECT lower(name), COUNT(*) FROM feature WHERE codebook=? AND level IN ('feature','variant') GROUP BY 1 HAVING COUNT(*)>1", (cbid,))]
print("\nduplicate names:", dups or "none")
fl = c.execute("SELECT verdict, standing, COUNT(*) k FROM flag WHERE codebook=? GROUP BY 1,2", (cbid,)).fetchall()
print("flags now:", [dict(r) for r in fl])
print("indistinct pairs still reported:")
for r in c.execute("SELECT a.name an, a.round ar, b.name bn, b.round br, f.standing FROM flag f JOIN feature a ON a.id=f.feature JOIN feature b ON b.id=f.other WHERE f.codebook=? AND f.verdict='indistinct' LIMIT 12", (cbid,)):
    print(f"   [{'standing' if r['standing'] else 'new'}] r{r['ar']} {r['an'][:40]:40} ~ r{r['br']} {r['bn'][:40]}")
print("\nstanding misfits, 8 at random (node <- wording : judge):")
sm = c.execute("SELECT a.name an, x.declaration d, f.note FROM flag f JOIN feature a ON a.id=f.feature JOIN realization x ON x.id=f.realization WHERE f.codebook=? AND f.verdict='misfit' AND f.standing=1", (cbid,)).fetchall()
for r in random.sample(sm, min(8, len(sm))):
    print(f"   {r['an'][:34]:34} <- {r['d'][:48]:48} : {str(r['note'])[:70]}")
print("\ncost of this library's calls:", c.execute("SELECT ROUND(SUM(COALESCE(billed,cost,0)),2) FROM call WHERE note LIKE ? AND at >= (SELECT at FROM codebook WHERE id=?)", (f"{corpus}:{kind}:%", cbid)).fetchone()[0])
