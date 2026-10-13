"""Build the index from a folder of .md/.txt/.rst/.py files (e.g. scraped docs saved by the browser tool).

    python -m rag.ingest ./docs_corpus --out data/index
    RAG_BACKEND=hashing python -m rag.ingest ./docs_corpus     # offline smoke test
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .index import HybridIndex
from .models import embedder_from_env, reranker_from_env
from .text import Doc, chunk_text

EXTS = {".md", ".txt", ".rst", ".py", ".js", ".ts", ".json"}


def collect(root: Path, max_chars: int, overlap: int) -> list[Doc]:
    docs: list[Doc] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in EXTS):
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, chunk in enumerate(chunk_text(text, max_chars=max_chars, overlap=overlap)):
            docs.append(Doc(id=f"{path.relative_to(root).as_posix()}#{i}", text=chunk, meta={"source": str(path.relative_to(root)), "chunk": i}))
    return docs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--out", default="data/index")
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=150)
    a = p.parse_args()
    docs = collect(Path(a.root), a.max_chars, a.overlap)
    if not docs:
        raise SystemExit(f"no indexable files under {a.root}")
    HybridIndex(embedder_from_env(), reranker_from_env()).build(docs).save(a.out)
    print(f"indexed {len(docs)} chunks -> {a.out}")


if __name__ == "__main__":
    main()
