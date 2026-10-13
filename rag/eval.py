"""Retrieval evaluation with ablations.

Eval file (JSONL), one query per line:
    {"query": "how do I resume a LangGraph run after a crash?", "relevant": ["doc-17", "doc-18"]}

``relevant`` are ids of chunks that actually answer the query (label them by hand or with a
reviewed LLM pass - this file is what makes a recall number defensible).

    python -m rag.eval --index data/index --queries data/eval.jsonl --k 5 10 20

Prints Recall@k and MRR@k for: bm25 / dense / hybrid / hybrid+rerank, so you can state exactly what
each stage contributes.  Recall@k here = fraction of queries with >=1 relevant doc in the top k
(hit-rate); --strict switches to fraction-of-relevant-docs-retrieved averaged over queries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .index import HybridIndex

MODES = {
    "bm25": dict(mode="bm25", rerank=False),
    "dense": dict(mode="dense", rerank=False),
    "hybrid": dict(mode="hybrid", rerank=False),
    "hybrid+rerank": dict(mode="hybrid", rerank=True),
}


def load_queries(path: str | Path) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        if not r.get("query") or not r.get("relevant"):
            raise ValueError(f"bad eval row: {r}")
    return rows


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int, strict: bool = False) -> float:
    top = set(ranked_ids[:k])
    if strict:
        return len(top & relevant) / len(relevant)
    return 1.0 if top & relevant else 0.0


def mrr_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    for i, d in enumerate(ranked_ids[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def evaluate(index: HybridIndex, queries: Iterable[dict], ks: list[int], strict: bool = False, modes: dict | None = None) -> dict:
    queries = list(queries)
    kmax = max(ks)
    results: dict[str, dict[str, float]] = {}
    for name, kwargs in (modes or MODES).items():
        if kwargs.get("rerank") and index.reranker is None:
            continue
        rec = {k: 0.0 for k in ks}
        mrr = {k: 0.0 for k in ks}
        for q in queries:
            ids = [h.doc.id for h in index.search(q["query"], k=kmax, **kwargs)]
            rel = set(q["relevant"])
            for k in ks:
                rec[k] += recall_at_k(ids, rel, k, strict)
                mrr[k] += mrr_at_k(ids, rel, k)
        n = len(queries)
        results[name] = {**{f"recall@{k}": rec[k] / n for k in ks}, **{f"mrr@{k}": mrr[k] / n for k in ks}}
    return {"n_queries": len(queries), "strict": strict, "results": results}


def format_report(rep: dict) -> str:
    rows = rep["results"]
    if not rows:
        return "no results"
    cols = list(next(iter(rows.values())))
    w = max(len(n) for n in rows) + 2
    lines = [f"queries: {rep['n_queries']}   recall mode: {'strict' if rep['strict'] else 'hit-rate'}", "".ljust(w) + "".join(c.rjust(11) for c in cols)]
    for name, m in rows.items():
        lines.append(name.ljust(w) + "".join(f"{m[c]:.3f}".rjust(11) for c in cols))
    return "\n".join(lines)


def main() -> None:
    from .models import embedder_from_env, reranker_from_env

    p = argparse.ArgumentParser()
    p.add_argument("--index", default="data/index")
    p.add_argument("--queries", required=True)
    p.add_argument("--k", type=int, nargs="+", default=[5, 10])
    p.add_argument("--strict", action="store_true")
    a = p.parse_args()
    idx = HybridIndex.load(a.index, embedder_from_env(), reranker_from_env())
    print(format_report(evaluate(idx, load_queries(a.queries), a.k, a.strict)))


if __name__ == "__main__":
    main()
