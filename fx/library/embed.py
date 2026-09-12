"""One vector per distinct wording, the retrieval behind the leftover sort. Computed once at collapse for the wordings
that have none, with a local sentence-transformers model (FX_EMBEDDER, default bge-large; FACET clustered with it),
stored in the `vector` table as float32. Tests pass an `encoder` callable and never load a model.

    embed(store, corpus, kind)               -> {"embedded": n, "model": ...}
    vectors(store, ids)                      -> (ids in order, float32 matrix, unit-normalised)
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

from ..corpus import corpus_id
from ..store import Store

EMBEDDER = os.environ.get("FX_EMBEDDER", "BAAI/bge-large-en-v1.5")
Encoder = Callable[[list[str]], np.ndarray]


def encoder(model: str = EMBEDDER) -> Encoder:
    """The local model, loaded on first use. Weights download into HF_HOME; point it at /data, not the home quota."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)

    def enc(texts: list[str]) -> np.ndarray:
        return np.asarray(m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    return enc


def embed(store: Store, corpus: str, kind: str, enc: Optional[Encoder] = None, model: str = EMBEDDER, batch: int = 512) -> dict:
    cid = corpus_id(store, corpus)
    rows = store.rows("SELECT r.id, r.polarity, r.declaration FROM realization r LEFT JOIN vector v ON v.realization=r.id WHERE r.corpus=? AND r.kind=? AND v.realization IS NULL ORDER BY r.id", (cid, kind))
    if not rows:
        return {"embedded": 0, "model": model}
    enc = enc or encoder(model)
    n = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        # polarity is part of identity, so it is part of the text the vector sees
        vecs = enc([f"{r['polarity']}: {r['declaration']}" for r in chunk])
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        with store.lock:
            store.con.executemany("INSERT OR REPLACE INTO vector (realization, model, dim, vec) VALUES (?,?,?,?)",
                                  [(int(r["id"]), model, int(vecs.shape[1]), vecs[k].astype(np.float32).tobytes()) for k, r in enumerate(chunk)])
            store.con.commit()
        n += len(chunk)
    return {"embedded": n, "model": model}


def vectors(store: Store, ids: list[int]) -> tuple[list[int], np.ndarray]:
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    out_ids, vecs = [], []
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in store.rows(f"SELECT realization, dim, vec FROM vector WHERE realization IN ({','.join('?' * len(chunk))})", chunk):
            out_ids.append(int(r["realization"])); vecs.append(np.frombuffer(r["vec"], dtype=np.float32, count=int(r["dim"])))
    return out_ids, (np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32))
