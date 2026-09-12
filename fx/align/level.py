"""The feature level: the units are the per-corpus features of every library of a kind (their cards), grouped by corpus,
placed on the seed codebook. The loop itself is fx.loop.engine."""
from __future__ import annotations

from ..loop import Level
from ..store import Store
from . import prompts as P
from .cards import card_text, cards
from .seed import seed_codebook

TAU = 0.84      # measured on the four guidance libraries (2026-09-11): same-name pairs sit at 0.90 and above, which is identity
                # of wording; 0.85 to 0.90 is mostly one instruction under different nouns; 0.80 to 0.85 mostly not


def feature_level(store: Store, kind: str) -> Level:
    cb = seed_codebook(store, kind)

    def units() -> list[dict]:
        return [c | {"text": card_text(c), "label": c["name"], "groups": {c["corpus"]}} for c in cards(store, kind)]

    def samples(u: dict) -> list[str]:
        """A member feature's wordings by support, its variants' included."""
        return [r["declaration"] for r in store.rows("SELECT r.declaration FROM membership m JOIN realization r ON r.id=m.unit WHERE m.kind='realization' AND m.node IN (SELECT id FROM feature WHERE id=? OR parent=?) "
                                                     "ORDER BY r.prompts DESC, r.n DESC LIMIT 6", (u["id"], u["id"]))]

    return Level(
        kind="feature", codebook=cb, units=units,
        render_units=P.render_cards, render_tree=P.render_globals,
        prompt_assign=lambda tr, us, shortlist: P.assign(tr, us, shortlist=shortlist, members_of=lambda n: _members_of(store, kind, cb, n)),
        prompt_name=lambda tr, ms: P.name(kind, tr, ms),
        prompt_judge=lambda node, ms, sm: P.judge(node, ms, sm),
        prompt_siblings=None, system=P.SYSTEM, member_samples=samples,
        node_vector="members", unit_prefix="F", node_prefix="S", min_members=2, min_groups=2, tau=TAU, measure_tau=None,
        named_min_members=2, named_min_groups=2, allow_variant=False, allow_new_group=False, batch=12, shortlist_k=5,
        aspects=tuple(P.ASPECTS_GUIDANCE if kind == "guidance" else P.ASPECTS_MATERIAL), label=f"seed:{kind}")


def _members_of(store: Store, kind: str, cb: int, node: dict) -> list[dict]:
    by = {c["id"]: c for c in cards(store, kind)}
    return [by[int(r["unit"])] for r in store.rows("SELECT unit FROM membership WHERE kind='feature' AND codebook=? AND node=?", (cb, node["id"])) if int(r["unit"]) in by]
