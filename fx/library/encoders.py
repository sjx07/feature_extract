"""The local embedding model (FX_EMBEDDER, default bge-large; FACET clustered with it), loaded on first use. Weights
download into HF_HOME: point it at /data, not the home quota. Tests pass an `encoder` callable and never load a model."""
from __future__ import annotations

import os
from typing import Callable

import numpy as np

EMBEDDER = os.environ.get("FX_EMBEDDER", "BAAI/bge-large-en-v1.5")
Encoder = Callable[[list[str]], np.ndarray]


def encoder(model: str = EMBEDDER) -> Encoder:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)

    def enc(texts: list[str]) -> np.ndarray:
        return np.asarray(m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    return enc
