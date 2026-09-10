"""The next codebook version from the current one, its leftovers and its flags, in one call: the only writer after the
cold start. A feature whose meaning is unchanged keeps its id (prev links the versions); a changed or merged one is new."""
from __future__ import annotations

import json
from typing import Optional

from ..corpus import corpus_domain
from ..llm import Client, ask
from ..store import Store
from ..util.ids import parse_ids
from ..util.jsonx import extract_object
from . import prompts as P
from .codebook import COLDSTART_MODEL, MAX_TOKENS, MIN_SUPPORT, REVISE_SCHEMA, flags, groups, latest, leftovers, new_codebook, write_codebook
from .collapse import realizations


def revise(store: Store, client: Client, corpus: str, kind: str, model: str = COLDSTART_MODEL, version: Optional[int] = None, min_support: int = MIN_SUPPORT,
           max_leftover: int = 600, domain: Optional[str] = None) -> dict:
    """The next version from the current one, its leftovers and its flags, in one call. Assign it afresh afterwards."""
    cbrow = latest(store, corpus, kind, version)
    if not cbrow:
        raise ValueError("no codebook")
    cb, cid = int(cbrow["id"]), int(cbrow["corpus"])
    tree = groups(store, cb)
    left = leftovers(store, cb, corpus, kind)[:max_leftover]
    fl = []
    for x in flags(store, cb):
        if x["verdict"] == "misfit":
            fl.append(f"misfit in F{x['feature']} {x['feature_name']}: R{x['realization']} \"{x['declaration']}\" ({x['note']})")
        elif x["verdict"] == "split":
            n = json.loads(x["note"] or "{}")
            fl.append(f"split suggested for F{x['feature']} {x['feature_name']}: {n.get('why', '')}; parts " + "; ".join(f"{p['name']} [{', '.join('R%d' % m for m in p['members'])}]" for p in n.get("parts", [])))
        else:
            fl.append(f"indistinct: F{x['feature']} {x['feature_name']} and F{x['other']} {x['other_name']} ({x['note']})")
    domain = domain or corpus_domain(store, cid, corpus)
    reply = ask(client, P.revise(kind, domain, tree, left, fl, min_support), model=model, stage="library", note=f"{corpus}:{kind}:revise:v{cbrow['version']}", system=P.SYSTEM, schema=REVISE_SCHEMA, max_tokens=MAX_TOKENS)
    obj = extract_object(reply, "groups")
    if obj is None:
        raise RuntimeError("revise reply was not a codebook (no 'groups' list)")
    all_r = {d["id"] for d in realizations(store, corpus, kind)}
    ncb, nversion = new_codebook(store, cid, kind, model, int(cbrow["round"]) + 1, str(obj.get("notes") or "")[:2000])
    w = write_codebook(store, ncb, obj, kind, all_r, old=tree)
    old_feats = {f["id"] for g in tree for f in g["features"]}
    retired = parse_ids(obj.get("retired"), old_feats)
    return {"corpus": corpus, "kind": kind, "codebook": ncb, "version": nversion, "leftover_seen": len(left), "flags_seen": len(fl), "retired": len(retired), **w, "notes": str(obj.get("notes") or "")[:500]}
