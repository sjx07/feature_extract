"""The library loop, written once against a Level (fx.loop.level), one module per step:

    embed         one vector per unit that has none
    assign        open units onto the tree's nodes or none: a batch against the whole tree, then the open ones
                  against the few nodes nearest by retrieval; a unit the judge removed carries the reason and its
                  node is offered back (putting it back settles the flag)
    judge         read-only: per node, members that do not fit and a split; per group, indistinct siblings; a flag
                  raised again after it was acted on is standing
    reopen        first-time flags send their member open with the reason; anchors never
    candidates    every open unit not yet looked at, with its nearest open units from other groups; a rejected seed is specific
    name          the fork: one call per neighbourhood, in parallel, each a proposal (variant, feature, or rejection); nothing written
    join          one call a round over every proposal beside the tree: new, same as an existing feature, or a duplicate of another
                  proposal; places the unplaced and founds a group where three agree; then the round writes once
    run_round     assign, judge, then reopen -> cluster -> name -> assign -> judge until settled

calls.py holds what the steps share to talk to the model, state.py what they read and write in the store.

Nothing written is rewritten: the tree only gains nodes (stamped with their round), a placed unit stays placed unless
the judge's flag reopens it, and only the open units are ever looked at again.
"""
from .assign import assign
from .calls import Stopped
from .join import join
from .judge import judge, reopen
from .level import Level
from .name import candidates, name, render_proposals
from .round import run_round
from .state import anchors, embed, members, node_vectors, nodes, open_units, tree, vectors

__all__ = ["Level", "Stopped", "anchors", "assign", "candidates", "embed", "judge", "members", "name", "node_vectors", "nodes", "render_proposals", "join", "open_units", "reopen", "run_round", "tree", "vectors"]
