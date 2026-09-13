"""Codebook version 1 from the corpus's realizations in one call. When they do not fit the model's context budget the call takes
the head of the support-ordered list and the codebook row says how many waited; the tail is singletons by construction, and the loop
(assign over everything, revise over the leftovers) is what reaches it."""
from __future__ import annotations

from typing import Optional

from fx.data.corpus import corpus_domain, corpus_id
from fx.core.llm import Client, ask
from fx.core.store import Store
from fx.core.util.jsonx import extract_object
from fx.ingest.induce import prompts as P
from fx.ingest.induce.codebook import COLDSTART_MODEL, COLDSTART_SCHEMA, CONTEXT_TOKENS, MAX_TOKENS, MIN_SUPPORT, fit, new_codebook, write_codebook
from fx.ingest.induce.collapse import realizations


def coldstart(store: Store, client: Client, corpus: str, kind: str, model: str = COLDSTART_MODEL, min_support: int = MIN_SUPPORT, domain: Optional[str] = None,
              context_tokens: int = CONTEXT_TOKENS, cache: bool = True) -> dict:
    cid = corpus_id(store, corpus)
    decl = realizations(store, corpus, kind)
    if not decl:
        raise ValueError(f"no {kind} realizations for {corpus}: run collapse, or decompose first")
    domain = domain or corpus_domain(store, cid, corpus)
    shown, waiting = fit(decl, context_tokens)
    reply = ask(client, P.coldstart(kind, domain, shown, min_support), model=model, stage="library", note=f"{corpus}:{kind}:coldstart", system=P.SYSTEM, schema=COLDSTART_SCHEMA, max_tokens=MAX_TOKENS, cache=cache)   # cache=False: a fresh listing, not the replay of an earlier one
    obj = extract_object(reply, "groups")
    if obj is None:
        raise RuntimeError("cold start reply was not a codebook (no 'groups' list)")
    cb, version = new_codebook(store, cid, kind, model, 0, notes=f"cold start over {len(shown)} of {len(decl)} declarations; {waiting} waited for the loop" if waiting else "")
    w = write_codebook(store, cb, obj, kind, {d["id"] for d in decl})
    return {"corpus": corpus, "kind": kind, "codebook": cb, "version": version, "declarations": len(decl), "shown": len(shown), "waiting": waiting, **w}
