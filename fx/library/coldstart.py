"""Codebook version 1 from every realization of the corpus in one call."""
from __future__ import annotations

from typing import Optional

from ..corpus import corpus_domain, corpus_id
from ..llm import Client, ask
from ..store import Store
from ..util.jsonx import extract_object
from . import prompts as P
from .codebook import COLDSTART_MODEL, COLDSTART_SCHEMA, MAX_TOKENS, MIN_SUPPORT, new_codebook, write_codebook
from .collapse import realizations


def coldstart(store: Store, client: Client, corpus: str, kind: str, model: str = COLDSTART_MODEL, min_support: int = MIN_SUPPORT, domain: Optional[str] = None) -> dict:
    cid = corpus_id(store, corpus)
    decl = realizations(store, corpus, kind)
    if not decl:
        raise ValueError(f"no {kind} realizations for {corpus}: run collapse, or decompose first")
    domain = domain or corpus_domain(store, cid, corpus)
    reply = ask(client, P.coldstart(kind, domain, decl, min_support), model=model, stage="library", note=f"{corpus}:{kind}:coldstart", system=P.SYSTEM, schema=COLDSTART_SCHEMA, max_tokens=MAX_TOKENS)
    obj = extract_object(reply, "groups")
    if obj is None:
        raise RuntimeError("cold start reply was not a codebook (no 'groups' list)")
    cb, version = new_codebook(store, cid, kind, model, 0)
    w = write_codebook(store, cb, obj, kind, {d["id"] for d in decl})
    return {"corpus": corpus, "kind": kind, "codebook": cb, "version": version, "declarations": len(decl), **w}
