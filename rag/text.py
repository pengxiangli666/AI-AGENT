"""Tokenisation and chunking shared by indexing and querying."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[㐀-䶿一-鿿぀-ヿ가-힯]+")
_SPLIT_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens; identifiers are also split (snake_case, CamelCase);
    CJK runs become character bigrams (plus the single char for 1-char runs), which is the
    standard dictionary-free way to make BM25 work on Chinese/Japanese text."""
    tokens: list[str] = []
    for m in _WORD.finditer(text):
        w = m.group()
        tokens.append(w.lower())
        parts = [p for chunk in w.split("_") for p in _SPLIT_CAMEL.split(chunk) if p]
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    for m in _CJK.finditer(text):
        run = m.group()
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass
class Doc:
    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking with character overlap (a paragraph longer than max_chars is split)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        while len(p) > max_chars:  # hard-split oversize paragraphs
            head, p = p[:max_chars], p[max_chars - overlap :]
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(head)
        if len(cur) + len(p) + 2 <= max_chars:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if cur and overlap else ""
            cur = f"{tail}\n\n{p}".strip() if tail else p
    if cur:
        chunks.append(cur)
    return chunks
