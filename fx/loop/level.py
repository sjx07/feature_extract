"""What differs between the two library levels, in one object; the loop in fx.loop is written against it.

A Level says what the units are (wordings of one corpus, or the features of every corpus), how a unit is shown to the
model, how units group for support (a wording's prompts; a card's corpus), how a node's vector is formed, which
codebook the loop writes into, and the prompt texts. Everything else, retrieval, batching, adjudication, standing
flags, the settle loop, is shared."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Level:
    kind: str                                   # the unit kind in membership/embedding: 'realization' or 'feature'
    codebook: int                               # the codebook the loop writes into
    units: Callable[[], list[dict]]             # every unit: {id, text, polarity, groups: set, support, ...}; the vector text under "text"
    render_units: Callable[..., str]            # (units, source=bool) -> the block the assigner/namer reads, ids as F<id>/R<id>
    render_tree: Callable[..., str]             # (tree, anchors=bool) -> the codebook as the model reads it
    prompt_assign: Callable[..., str]           # (tree, units, shortlist) -> str
    prompt_name: Callable[..., str]             # (tree, members) -> str
    prompt_judge: Callable[..., str]            # (node, members, samples) -> str
    prompt_siblings: Optional[Callable[..., str]]   # (group, samples) -> str, or None to skip the sibling check
    system: str
    member_samples: Callable[[dict], list[str]]     # what the judge sees of a member beyond its line
    prompt_group: Optional[Callable[..., str]] = None   # (groups, unplaced features) -> str; None: the groups are fixed (the seed)
    node_vector: str = "anchors"                # 'anchors': mean of the node's example units; 'members': mean of its members
    unit_prefix: str = "R"                      # how the unit id is written in prompts
    node_prefix: str = "F"
    named_min_members: int = 2                  # a named node needs this many accepted members ...
    named_min_groups: int = 1                   # ... from this many groups
    group_min_features: int = 3                 # a new group needs this many unplaced features that share one purpose
    allow_variant: bool = True                  # may the namer place a cluster under a feature as a variant
    batch: int = 20
    shortlist_k: int = 4
    neighbourhood: int = 5                      # a candidate is a seed unit and its k nearest open units from other groups; no threshold
    aspects: tuple = ()
    label: str = ""                             # for logs and notes: 'text2sql guidance', 'seed guidance'
