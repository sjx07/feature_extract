"""The wording level: the units are one corpus's realizations (distinct declarations), grouped by the prompts they occur
in, placed on the corpus's codebook. Everything the loop needs to know about this level is here; the loop itself is
fx.loop.engine."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

import numpy as np

from ..corpus import corpus_domain
from ..loop import Level, vectors
from ..store import Store
from . import prompts as P
from .codebook import BATCH, MIN_SUPPORT

TAU_DEFAULT, TAU_MIN, TAU_MAX = 0.78, 0.6, 0.92


def wording_level(store: Store, codebook: Optional[int] = None, corpus: Optional[str] = None, kind: Optional[str] = None) -> Level:
    """From a codebook id, or from corpus and kind when there is no codebook yet (embedding can run before the cold start)."""
    if codebook:
        cb = store.one("SELECT c.*, k.name corpus_name FROM codebook c JOIN corpus k ON k.id=c.corpus WHERE c.id=?", (codebook,))
        cid, kind, corpus = int(cb["corpus"]), cb["kind"], cb["corpus_name"]
    else:
        from ..corpus import corpus_id
        cid, codebook = corpus_id(store, corpus), 0
    domain = corpus_domain(store, cid, corpus)

    def units() -> list[dict]:
        prompts_of: dict[int, set] = defaultdict(set)
        for r in store.rows("SELECT r.realization, r.prompt FROM reading r JOIN realization x ON x.id=r.realization WHERE x.corpus=? AND x.kind=?", (cid, kind)):
            prompts_of[int(r["realization"])].add(r["prompt"])
        out = []
        for r in store.rows("SELECT * FROM realization WHERE corpus=? AND kind=? ORDER BY prompts DESC, n DESC, id", (cid, kind)):
            d = dict(r) | {"conditions": json.loads(r["conditions"] or "[]"), "domain_terms": json.loads(r["domain_terms"] or "[]")}
            d |= {"text": f"{d['polarity']}: {d['declaration']}", "label": d["declaration"], "groups": prompts_of.get(d["id"], set()), "support": int(d["prompts"])}
            out.append(d)
        return out

    def measure_tau() -> Optional[float]:
        """The similarity at which known-same pairs (a node's anchors) mostly count as neighbours: the 25th percentile of
        anchor-pair cosines, clamped; None with fewer than 10 pairs."""
        sims = []
        for f in store.rows("SELECT examples FROM feature WHERE codebook=? AND level IN ('feature','variant')", (codebook,)):
            ex = json.loads(f["examples"] or "[]")
            if len(ex) >= 2:
                ids, m = vectors(store, "realization", ex)
                sims += [float(m[i] @ m[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
        return float(min(TAU_MAX, max(TAU_MIN, np.quantile(sims, 0.25)))) if len(sims) >= 10 else None

    return Level(
        kind="realization", codebook=codebook, units=units,
        render_units=P.render_declarations, render_tree=P.render_codebook,
        prompt_assign=lambda tr, us, shortlist: P.assign(kind, tr, us, shortlist=shortlist),
        prompt_name=lambda tr, ms: P.name(kind, domain, tr, ms),
        prompt_judge=lambda node, ms, samples: P.judge_members(node, ms),
        prompt_siblings=P.judge_siblings, system=P.SYSTEM, member_samples=lambda u: [],
        node_vector="anchors", unit_prefix="R", node_prefix="F", min_members=MIN_SUPPORT, min_groups=2, tau=TAU_DEFAULT, measure_tau=measure_tau,
        named_min_members=2, named_min_groups=1, allow_variant=True, batch=BATCH, shortlist_k=4,
        aspects=tuple(P.ASPECTS_GUIDANCE if kind == "guidance" else P.ASPECTS_MATERIAL), label=f"{corpus}:{kind}")
