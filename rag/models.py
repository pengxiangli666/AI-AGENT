"""Embedder / reranker backends.

Production: BGE-M3 (dense) + bge-reranker via sentence-transformers.
Offline / tests: ``HashingEmbedder`` and ``LexicalOverlapReranker`` - deterministic, dependency
free, and *not* semantic.  Numbers produced with them say nothing about BGE-M3 quality;
always run rag.eval with the real models before quoting any recall figure.
"""
from __future__ import annotations

import hashlib
import os
from typing import Protocol, Sequence

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...  # (n, dim) float32, L2-normalised


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype("float32")


class BGEM3Embedder:
    """BAAI/bge-m3 dense embeddings (1024-d, multilingual)."""

    def __init__(self, model_name: str | None = None, device: str | None = None, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.name = model_name or os.getenv("EMBED_MODEL", "BAAI/bge-m3")
        self._m = SentenceTransformer(self.name, device=device)
        self._m.max_seq_length = int(os.getenv("EMBED_MAX_LEN", "512"))
        self.dim = int(self._m.get_sentence_embedding_dimension())
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        v = self._m.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype="float32")


class HashingEmbedder:
    """Feature-hashed bag of tokens. Deterministic, offline. Lexical, not semantic."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0  # signed hashing trick
        return _l2(out)


class CrossEncoderReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder."""

    def __init__(self, model_name: str | None = None, device: str | None = None, batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.name = model_name or os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        self._m = CrossEncoder(self.name, device=device, max_length=512)
        self.batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        pairs = [(query, p) for p in passages]
        return [float(s) for s in self._m.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)]


class LexicalOverlapReranker:
    """Cheap stand-in: fraction of query tokens present in the passage."""

    name = "lexical-overlap"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(passages)
        return [len(q & set(tokenize(p))) / len(q) for p in passages]


def embedder_from_env() -> Embedder:
    backend = os.getenv("RAG_BACKEND", "bge")
    if backend == "hashing":
        return HashingEmbedder()
    return BGEM3Embedder()


def reranker_from_env() -> Reranker | None:
    backend = os.getenv("RAG_BACKEND", "bge")
    if backend == "hashing":
        return LexicalOverlapReranker()
    if os.getenv("RAG_RERANK", "1") == "0":
        return None
    return CrossEncoderReranker()
