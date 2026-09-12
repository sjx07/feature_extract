"""The feature level: the units are the per-corpus features of every library of a kind (their cards), grouped by corpus,
placed on the seed codebook. The loop itself is fx.loop."""
from __future__ import annotations

from typing import Optional

from ..loop import Level, render_proposals
from ..store import Store
from . import prompts as P
from .cards import card_text, cards
from .seed import seed_codebook



def feature_level(store: Store, kind: str) -> Level:
    cb = seed_codebook(store, kind)

    def units() -> list[dict]:
        return [c | {"text": card_text(c), "label": c["name"], "groups": {c["corpus"]}} for c in cards(store, kind)]

    def samples(u: dict) -> list[str]:
        """A member feature's wordings by support, its variants' included."""
        return [r["declaration"] for r in store.rows("SELECT r.declaration FROM membership m JOIN realization r ON r.id=m.unit WHERE m.kind='realization' AND m.node IN (SELECT id FROM feature WHERE id=? OR parent=?) "
                                                     "ORDER BY r.prompts DESC, r.n DESC LIMIT 6", (u["id"], u["id"]))]

    cache: dict = {}                        # the cards, built once per level: an assign renders every global's members in every batch prompt
    return Level(
        kind="feature", codebook=cb, units=units,
        render_units=P.render_cards, render_tree=P.render_globals,
        prompt_assign=lambda tr, us, shortlist: P.assign(tr, us, shortlist=shortlist, members_of=lambda n: _members_of(store, kind, cb, n, cache)),
        prompt_name=lambda tr, ms: P.name(kind, tr, ms),
        prompt_judge=lambda node, ms, sm: P.judge(node, ms, sm),
        prompt_siblings=lambda g, sm: P.judge_siblings(g, sm), system=P.SYSTEM, member_samples=samples,
        prompt_join=lambda tr, ps, us, qs: P.join_prompt(tr, render_proposals(ps, "F"), qs), groups_fixed=True,
        node_vector="members", unit_prefix="F", node_prefix="S",
        named_min_members=2, named_min_groups=2, allow_variant=False, batch=12, shortlist_k=5, neighbourhood=5,
        aspects=tuple(P.ASPECTS_GUIDANCE if kind == "guidance" else P.ASPECTS_MATERIAL), label=f"seed:{kind}")


def _members_of(store: Store, kind: str, cb: int, node: dict, cache: Optional[dict] = None) -> list[dict]:
    """The cards on a global. Rebuilding every card per call took the seed's first assign (40 globals × 100 batches) to
    hours of CPU with no call in flight; the cache holds them for the life of the level."""
    if cache is None or "by" not in cache:
        by = {c["id"]: c for c in cards(store, kind)}
        if cache is not None:
            cache["by"] = by
    else:
        by = cache["by"]
    return [by[int(r["unit"])] for r in store.rows("SELECT unit FROM membership WHERE kind='feature' AND codebook=? AND node=?", (cb, node["id"])) if int(r["unit"]) in by]
