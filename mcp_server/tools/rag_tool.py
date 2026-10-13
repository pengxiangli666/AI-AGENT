"""Expose the hybrid index to the agent as an MCP tool."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from mcp_server.registry import mcp_tool


@lru_cache(maxsize=1)
def _index():
    from rag.index import HybridIndex
    from rag.models import embedder_from_env, reranker_from_env

    path = os.getenv("RAG_INDEX_DIR", "./data/index")
    return HybridIndex.load(path, embedder_from_env(), reranker_from_env())


@mcp_tool
async def search_docs(query: str, k: int = 5, mode: str = "hybrid") -> dict[str, Any]:
    """Search the indexed documentation / codebase knowledge with hybrid BM25 + dense retrieval and reranking.

    Args:
        query: Natural-language question or identifier to look up.
        k: Number of passages to return (1-20).
        mode: "hybrid" (default), "bm25" or "dense".
    """
    if mode not in ("hybrid", "bm25", "dense"):
        raise ValueError("mode must be hybrid, bm25 or dense")
    k = max(1, min(int(k), 20))
    try:
        idx = _index()
    except FileNotFoundError as e:
        raise RuntimeError("no index found - build one with: python -m rag.ingest <docs_dir>") from e
    hits = idx.search(query, k=k, mode=mode)
    return {
        "results": [
            {"id": h.doc.id, "score": round(h.score, 4), "source": h.doc.meta.get("source"), "ranks": h.ranks, "text": h.doc.text[:1500]}
            for h in hits
        ]
    }
