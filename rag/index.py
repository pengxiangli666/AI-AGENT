"""Hybrid retrieval: BM25 (lexical) + dense vectors (Faiss), fused with RRF, then reranked."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .models import Embedder, Reranker
from .text import Doc, tokenize

RRF_K = 60  # standard constant from the original RRF paper


@dataclass
class Hit:
    doc: Doc
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # which retrievers found it, at what rank (1-based)


def rrf_fuse(rankings: dict[str, list[int]], weights: dict[str, float] | None = None, k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over ranked lists of doc indices. Returns [(doc_idx, score)] best first."""
    scores: dict[int, float] = {}
    for name, ranking in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridIndex:
    def __init__(self, embedder: Embedder, reranker: Reranker | None = None):
        self.embedder = embedder
        self.reranker = reranker
        self.docs: list[Doc] = []
        self._bm25: BM25Okapi | None = None
        self._tokens: list[list[str]] = []
        self._faiss: faiss.Index | None = None

    # ------------------------------------------------------------------ build / persist
    def build(self, docs: Sequence[Doc], batch: int = 64) -> "HybridIndex":
        ids = [d.id for d in docs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate doc ids")
        self.docs = list(docs)
        self._tokens = [tokenize(d.text) for d in self.docs]
        self._bm25 = BM25Okapi(self._tokens) if self.docs else None
        vecs = [self.embedder.embed([d.text for d in self.docs[i : i + batch]]) for i in range(0, len(self.docs), batch)]
        mat = np.vstack(vecs) if vecs else np.zeros((0, self.embedder.dim), dtype="float32")
        self._faiss = faiss.IndexFlatIP(self.embedder.dim)  # exact inner product == cosine (normalised)
        self._faiss.add(mat)
        return self

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "docs.jsonl").open("w", encoding="utf-8") as f:
            for doc in self.docs:
                f.write(json.dumps({"id": doc.id, "text": doc.text, "meta": doc.meta}, ensure_ascii=False) + "\n")
        faiss.write_index(self._faiss, str(d / "vectors.faiss"))
        (d / "meta.json").write_text(json.dumps({"embedder": self.embedder.name, "dim": self.embedder.dim, "n": len(self.docs)}))

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder, reranker: Reranker | None = None) -> "HybridIndex":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        if meta["embedder"] != embedder.name or meta["dim"] != embedder.dim:
            raise ValueError(f"index was built with {meta['embedder']} (dim {meta['dim']}), got {embedder.name} (dim {embedder.dim})")
        idx = cls(embedder, reranker)
        idx.docs = [Doc(**json.loads(line)) for line in (d / "docs.jsonl").read_text(encoding="utf-8").splitlines() if line]
        idx._tokens = [tokenize(x.text) for x in idx.docs]  # BM25 is cheap to rebuild; avoids pickling
        idx._bm25 = BM25Okapi(idx._tokens) if idx.docs else None
        idx._faiss = faiss.read_index(str(d / "vectors.faiss"))
        return idx

    # ------------------------------------------------------------------ retrieval
    def _bm25_rank(self, query: str, n: int) -> list[int]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")[:n]
        return [int(i) for i in order if scores[i] > 0]

    def _dense_rank(self, query: str, n: int) -> list[int]:
        if not self.docs:
            return []
        q = self.embedder.embed([query])
        _, ids = self._faiss.search(q, min(n, len(self.docs)))
        return [int(i) for i in ids[0] if i >= 0]

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        mode: str = "hybrid",          # "bm25" | "dense" | "hybrid"
        candidates: int = 50,          # per-retriever depth fed into fusion
        rerank: bool = True,
        rerank_top: int = 30,          # how many fused results the cross-encoder re-scores
        weights: dict[str, float] | None = None,
    ) -> list[Hit]:
        rankings: dict[str, list[int]] = {}
        if mode in ("bm25", "hybrid"):
            rankings["bm25"] = self._bm25_rank(query, candidates)
        if mode in ("dense", "hybrid"):
            rankings["dense"] = self._dense_rank(query, candidates)
        if not rankings:
            raise ValueError(f"unknown mode {mode!r}")

        fused = rrf_fuse(rankings, weights)
        rank_of = {name: {idx: r for r, idx in enumerate(lst, start=1)} for name, lst in rankings.items()}
        hits = [Hit(self.docs[i], s, {n: rank_of[n][i] for n in rank_of if i in rank_of[n]}) for i, s in fused]

        if rerank and self.reranker and hits:
            head, tail = hits[:rerank_top], hits[rerank_top:]
            scores = self.reranker.score(query, [h.doc.text for h in head])
            for h, s in zip(head, scores):
                h.score = float(s)
            head.sort(key=lambda h: -h.score)
            hits = head + tail
        return hits[:k]
