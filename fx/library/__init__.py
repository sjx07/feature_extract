"""Stage 2, the feature library of a corpus, one per kind (guidance, material). Five steps over the store:

    collapse(store, corpus, kind)                       readings -> realizations (distinct declarations), no calls
    coldstart(store, client, corpus, kind, model)       one call over every declaration -> codebook version 1
    assign(store, client, corpus, kind, model)          batches of declarations on the latest version -> assignments
    judge(store, client, corpus, kind, model)           read-only coherence: misfits, splits, indistinct siblings -> flags
    revise(store, client, corpus, kind, model)          leftovers + flags -> the next version (the only writer after cold start)

A version is a full snapshot; assign runs from scratch on each version, so no assignment depends on the path that
produced it. The anchor agreement of a version is the share of its features' own example declarations that assign
puts back on them; it is measured, stored on the codebook row, and shown.
"""
from .assign import assign
from .codebook import BATCH, COLDSTART_MODEL, KINDS, MIN_SUPPORT, anchors, flags, groups, latest, leftovers, members, status
from .coldstart import coldstart
from .collapse import collapse, realizations
from .judge import judge
from .preview import preview
from .revise import revise

__all__ = ["BATCH", "COLDSTART_MODEL", "KINDS", "MIN_SUPPORT", "anchors", "assign", "coldstart", "collapse", "flags", "groups", "judge", "latest",
           "leftovers", "members", "preview", "realizations", "revise", "status"]
