"""Stage 2, the feature library of a corpus, one per kind (guidance, material). A tree that only grows:

    collapse(store, corpus, kind)                     readings -> realizations (distinct wordings), no calls
    embed(store, corpus, kind)                        one vector per realization, a local model
    coldstart(store, client, corpus, kind, model)     one call over every wording -> the codebook (groups, features)
    assign(store, client, corpus, kind, model)        wordings onto the tree's nodes, or open; then the retrieval shortlist pass
    judge(store, client, corpus, kind, model)         read-only coherence: misfits, splits, indistinct siblings -> flags
    candidates(store, cb, corpus, kind)               cluster the open wordings; the neighbourless are specific
    name(store, client, corpus, kind, cb, clusters)   one call per cluster -> a variant, a feature, or a rejection
    run_round(...)                                    the loop above until no candidate is left

Anchor agreement, measured after every assign, is the share of the nodes' own example wordings that landed back on them.
"""
from .assign import assign
from .cluster import candidates, threshold
from .codebook import BATCH, COLDSTART_MODEL, KINDS, MIN_SUPPORT, anchors, flags, groups, latest, leftovers, members, nodes, status
from .coldstart import coldstart
from .collapse import collapse, realizations
from .embed import embed, vectors
from .judge import judge
from .name import name
from .preview import preview
from .reopen import reopen
from .round import run_round

__all__ = ["BATCH", "COLDSTART_MODEL", "KINDS", "MIN_SUPPORT", "anchors", "assign", "candidates", "coldstart", "collapse", "embed", "flags", "groups", "judge", "latest",
           "leftovers", "members", "name", "nodes", "preview", "realizations", "reopen", "run_round", "status", "threshold", "vectors"]
