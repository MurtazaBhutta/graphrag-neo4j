"""Embeddings. One local model (BAAI/bge-small-en-v1.5 via fastembed, ONNX on CPU) for both
pipelines, so the benchmark compares retrieval strategies rather than embedding models."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384


class Embedder(Protocol):
    dimensions: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    dimensions = DIMENSIONS

    def __init__(self, cache_dir: str | os.PathLike | None = None):
        # Imported lazily: importing fastembed is slow and the first use downloads ~70 MB.
        from fastembed import TextEmbedding

        self._cache = Path(cache_dir or os.environ.get("GRAPHRAG_CACHE", ".cache"))
        self._model = TextEmbedding(MODEL, cache_dir=str(self._cache / "models"))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Cached on disk by content hash, so re-loading the same world skips the slow part."""
        key = hashlib.sha256("\x1e".join(texts).encode()).hexdigest()[:20]
        path = self._cache / f"embeddings-{key}.npy"
        if path.exists():
            return np.load(path)
        vectors = np.asarray(
            list(self._model.passage_embed(texts, batch_size=256)), dtype=np.float32
        )
        self._cache.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        # bge models expect an instruction prefix on queries; query_embed adds it.
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)


class HashEmbedder:
    """Deterministic stand-in for unit tests: no model, no download. Not semantically useful."""

    dimensions = 32

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_query(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
        vector = np.random.default_rng(seed).normal(size=self.dimensions).astype(np.float32)
        return vector / np.linalg.norm(vector)
