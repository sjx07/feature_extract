"""The next codebook version: SHARPEN (one call: the codebook and the judge's flags -> the same features, sharpened, merged
or retired) and then GROW (calls over the leftovers in blocks by head -> new features with support, appended to the
version). Growth stops when a block's proposals cover under `min_grow_gain` of the leftovers it saw. The two are
separate because one call doing both was conservative: the pilot's single revise added 20 features from 1,610
leftovers while its sharpening pushed wordings out, for a net gain of 400.

A feature whose meaning is unchanged keeps its id (prev links the versions) and its anchors; a changed or merged one
is new."""
from __future__ import annotations

import json
from typing import Optional

from ..corpus import corpus_domain
from ..llm import Client, ask
from ..store import Store
from ..util.ids import parse_ids
from ..util.jsonx import extract_object
from . import prompts as P
from .codebook import COLDSTART_MODEL, CONTEXT_TOKENS, GROW_SCHEMA, MAX_TOKENS, MIN_SUPPORT, REVISE_SCHEMA, add_grown, fit, flags, groups, latest, leftovers, new_codebook, write_codebook
from .collapse import realizations


MIN_GROW_GAIN = 0.02     # a grow block whose proposals cover under this share of the leftovers it saw ends the growth
GROW_BLOCK_TOKENS = 30_000


def revise(store: Store, client: Client, corpus: str, kind: str, model: str = COLDSTART_MODEL, version: Optional[int] = None, min_support: int = MIN_SUPPORT,
           context_tokens: int = CONTEXT_TOKENS, domain: Optional[str] = None, grow: bool = True, min_grow_gain: float = MIN_GROW_GAIN) -> dict:
    cbrow = latest(store, corpus, kind, version)
    if not cbrow:
        raise ValueError("no codebook")
    cb, cid = int(cbrow["id"]), int(cbrow["corpus"])
    tree = groups(store, cb)
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
    all_r = {d["id"] for d in realizations(store, corpus, kind)}
    # 1. sharpen: the same features, revised where the flags warrant it
    reply = ask(client, P.sharpen(kind, domain, tree, fl), model=model, stage="library", note=f"{corpus}:{kind}:sharpen:v{cbrow['version']}", system=P.SYSTEM, schema=REVISE_SCHEMA, max_tokens=MAX_TOKENS)
    obj = extract_object(reply, "groups")
    if obj is None:
        raise RuntimeError("sharpen reply was not a codebook (no 'groups' list)")
    ncb, nversion = new_codebook(store, cid, kind, model, int(cbrow["round"]) + 1, str(obj.get("notes") or "")[:2000])
    w = write_codebook(store, ncb, obj, kind, all_r, old=tree)
    old_feats = {f["id"] for g in tree for f in g["features"]}
    out = {"corpus": corpus, "kind": kind, "codebook": ncb, "version": nversion, "flags_seen": len(fl), "retired": len(parse_ids(obj.get("retired"), old_feats)), **w,
           "grow_calls": 0, "grown": 0, "grow_dropped": 0, "leftover_seen": 0, "leftover_covered": 0, "notes": str(obj.get("notes") or "")[:500]}
    if not grow:
        return out
    # 2. grow: the old version's leftovers in blocks by head, each against the new version's codebook as it stands
    left = leftovers(store, cb, corpus, kind)
    left.sort(key=lambda r: (r["polarity"], r.get("head") or "", -r["prompts"], -r["n"]))
    blocks: list[list[dict]] = []
    while left:
        chunk, _ = fit(left, GROW_BLOCK_TOKENS, overhead_tokens=1500)
        if not chunk:
            break
        blocks.append(chunk); left = left[len(chunk):]
    for k, block in enumerate(blocks):
        reply = ask(client, P.grow(kind, domain, groups(store, ncb), block, min_support), model=model, stage="library", note=f"{corpus}:{kind}:grow:v{nversion}:{k}", system=P.SYSTEM, schema=GROW_SCHEMA, max_tokens=MAX_TOKENS)
        obj = extract_object(reply, "features")
        out["grow_calls"] += 1; out["leftover_seen"] += len(block)
        if obj is None:
            continue
        g = add_grown(store, ncb, obj["features"], kind, all_r, min_support)
        out["grown"] += g["added"]; out["grow_dropped"] += g["dropped"]; out["leftover_covered"] += g["covered"]
        if g["covered"] < min_grow_gain * len(block):
            out["grow_stopped"] = f"block {k + 1} of {len(blocks)} covered {g['covered']} of {len(block)}"
            break
    out["features"] = int(store.one("SELECT COUNT(*) k FROM feature WHERE codebook=? AND level='feature'", (ncb,))["k"])
    return out
