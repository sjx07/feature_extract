"""One vector per per-corpus feature card, the retrieval behind cross-corpus candidates. The same local model as the
wordings' vectors; stored in `fvector`; tests pass an encoder."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..library.embed import EMBEDDER, Encoder, encoder
from ..store import Store
from .cards import card_text, cards


def embed(store: Store, kind: str, enc: Optional[Encoder] = None, model: str = EMBEDDER) -> dict:
    have = {int(r["feature"]) for r in store.rows("SELECT feature FROM fvector")}
    todo = [c for c in cards(store, kind) if c["id"] not in have]
    if not todo:
        return {"embedded": 0, "model": model}
    enc = enc or encoder(model)
    vecs = enc([card_text(c) for c in todo])
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    with store.lock:
        store.con.executemany("INSERT OR REPLACE INTO fvector (feature, model, dim, vec) VALUES (?,?,?,?)",
                              [(c["id"], model, int(vecs.shape[1]), vecs[k].astype(np.float32).tobytes()) for k, c in enumerate(todo)])
        store.con.commit()
    return {"embedded": len(todo), "model": model}


def vectors(store: Store, ids: list[int]) -> tuple[list[int], np.ndarray]:
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    out_ids, vecs = [], []
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in store.rows(f"SELECT feature, dim, vec FROM fvector WHERE feature IN ({','.join('?' * len(chunk))})", chunk):
            out_ids.append(int(r["feature"])); vecs.append(np.frombuffer(r["vec"], dtype=np.float32, count=int(r["dim"])))
    return out_ids, (np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32))
