"""What differs between the two library levels, in one object; the engine in fx.loop.engine is written against it.

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
    node_vector: str = "anchors"                # 'anchors': mean of the node's example units; 'members': mean of its members
    unit_prefix: str = "R"                      # how the unit id is written in prompts
    node_prefix: str = "F"
    min_members: int = 3                        # a candidate cluster needs this many units ...
    min_groups: int = 2                         # ... from this many distinct groups (prompts, or corpora)
    tau: float = 0.78                           # the neighbour threshold when none is measured or given
    measure_tau: Optional[Callable[[], Optional[float]]] = None   # e.g. from the anchors; None -> use tau
    named_min_members: int = 2                  # a named node needs this many accepted members ...
    named_min_groups: int = 1                   # ... from this many groups
    allow_variant: bool = True                  # may the namer place a cluster under a feature as a variant
    allow_new_group: bool = True                # may the namer create a group (the seed's groups are the aspects, fixed at birth)
    batch: int = 20
    shortlist_k: int = 4
    neighbourhood: int = 0                      # >0: candidates are neighbourhoods (a seed unit and its k nearest from other groups),
                                                # no threshold; the namer decides and a rejected seed is specific. 0: threshold clusters
    aspects: tuple = ()
    label: str = ""                             # for logs and notes: 'text2sql guidance', 'seed guidance'
