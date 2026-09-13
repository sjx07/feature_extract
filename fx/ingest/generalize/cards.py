"""The units of alignment: one card per per-corpus feature, from every corpus's latest codebook of a kind. A card is
what the aligner reads and what the vector is made of: the feature's name, definition, polarity, anchors, the corpus
it comes from and its support there. Variants are not aligned; they stay under their feature, so the hierarchy is
global feature > per-corpus feature > variant."""
from __future__ import annotations

import json
from typing import Optional

from fx.core.store import Store

SEED = "seed"        # the name of the seed's line in jobs and history; the seed codebook itself is the one with scope 'seed'


def seed_codebook_id(store: Store, kind: str) -> Optional[int]:
    r = store.one("SELECT id FROM codebook WHERE scope='seed' AND kind=?", (kind,))
    return int(r["id"]) if r else None


def libraries(store: Store, kind: str) -> list[dict]:
    """The latest codebook per corpus of this kind, the seed excluded."""
    return [dict(r) for r in store.rows("SELECT c.id codebook, c.corpus, k.name corpus_name, c.version FROM codebook c JOIN corpus k ON k.id=c.corpus WHERE c.kind=? AND c.scope='corpus' "
                                        "AND c.id = (SELECT MAX(id) FROM codebook x WHERE x.corpus=c.corpus AND x.kind=c.kind) ORDER BY k.name", (kind,))]


def cards(store: Store, kind: str, corpus: Optional[str] = None) -> list[dict]:
    libs = libraries(store, kind)
    if corpus:
        libs = [l for l in libs if l["corpus_name"] == corpus]
    out = []
    for lib in libs:
        cb = lib["codebook"]
        sup = {int(r["node"]): (int(r["prompts"]), int(r["k"])) for r in
               store.rows("SELECT m.node, SUM(r.prompts) prompts, COUNT(*) k FROM membership m JOIN realization r ON r.id=m.unit WHERE m.kind='realization' AND m.codebook=? AND m.node IS NOT NULL GROUP BY m.node", (cb,))}
        ex_text = {int(r["id"]): r["declaration"] for r in store.rows("SELECT id, declaration FROM realization WHERE corpus=? AND kind=?", (lib["corpus"], kind))}
        for f in store.rows("SELECT f.*, g.name grp FROM feature f JOIN feature g ON g.id=f.parent WHERE f.codebook=? AND f.level='feature' ORDER BY f.id", (cb,)):
            ex = json.loads(f["examples"] or "[]")
            vs = [dict(v) for v in store.rows("SELECT id, name FROM feature WHERE parent=? AND level='variant'", (f["id"],))]
            prompts, k = sup.get(int(f["id"]), (0, 0))
            for v in vs:
                p2, k2 = sup.get(int(v["id"]), (0, 0)); prompts += p2; k += k2
            out.append({"id": int(f["id"]), "corpus": lib["corpus_name"], "codebook": cb, "name": f["name"], "definition": f["definition"] or "", "polarity": f["polarity"] or "require",
                        "group": f["grp"], "anchors": [ex_text[e] for e in ex if e in ex_text], "support": prompts, "wordings": k, "variants": [v["name"] for v in vs]})
    return out


def card_text(c: dict) -> str:
    """What the vector sees: polarity, name, definition and anchors; never the corpus or the domain's own nouns beyond what the wording carries."""
    return f"{c['polarity']}: {c['name']}. {c['definition']} e.g. " + " | ".join(c["anchors"][:3])


def render_cards(rows: list[dict], ids: bool = True) -> str:
    """Cards as the model reads them: F<id> [corpus] (polarity) name: definition · e.g. anchors · support."""
    out = []
    for c in rows:
        head = f"F{c['id']} " if ids else ""
        out.append(f"{head}[{c['corpus']}] ({c['polarity']}) {c['name']}: {c['definition']}" + (f"\n      e.g. {' | '.join(a[:90] for a in c['anchors'][:3])}" if c["anchors"] else "")
                   + f"\n      {c['support']} prompts, {c['wordings']} wordings" + (f"; variants: {', '.join(c['variants'][:4])}" if c["variants"] else "")
                   + (f"\n      the judge removed this from S{c['judged'][0]}: {c['judged'][1] or 'no reason given'}. Put it back only if the judge is wrong; else the global it is an instance of, or null." if c.get("judged") else ""))
    return "\n".join(out) if out else "(none)"
