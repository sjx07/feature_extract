"""Stage 3, the seed library: the per-corpus libraries aligned into global features, the same loop as stage 2 one level up.

    cards(store, kind)                 the units: one card per per-corpus feature (name, definition, anchors, support)
    embed(store, kind)                 one vector per card
    assign(store, client, kind)        open cards onto the nearest globals, or none
    candidates(store, kind)            open cards that neighbour cards from other corpora; the rest domain-specific
    name(store, client, kind, ...)     one call per candidate: one global feature, or a rejection
    judge(store, client, kind)         read-only: members whose wordings give another instruction
    run_round(store, client, kind)     the loop until settled
"""
from .assign import assign
from .cards import cards, libraries
from .cluster import candidates, threshold
from .embed import embed
from .judge import judge, reopen
from .name import name
from .round import run_round
from .seed import globals_, open_cards, seed_codebook, status

__all__ = ["assign", "candidates", "cards", "embed", "globals_", "judge", "libraries", "name", "open_cards", "reopen", "run_round", "seed_codebook", "status", "threshold"]
