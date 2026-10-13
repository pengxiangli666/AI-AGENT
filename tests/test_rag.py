import numpy as np
import pytest

from rag.eval import evaluate, format_report, mrr_at_k, recall_at_k
from rag.index import HybridIndex, rrf_fuse
from rag.models import HashingEmbedder, LexicalOverlapReranker
from rag.text import Doc, chunk_text, tokenize

# ---------------------------------------------------------------- a tiny "semantic" embedder for tests
CONCEPTS = [
    {"car", "automobile", "vehicle"},
    {"fix", "repair", "mend"},
    {"crash", "failure", "outage"},
    {"resume", "continue", "restart"},
    {"database", "postgres", "sql"},
]


class ConceptEmbedder:
    """Maps synonyms onto shared dimensions: finds meaning BM25 cannot, knows no identifiers."""

    name, dim = "concept-test", len(CONCEPTS)

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                for j, group in enumerate(CONCEPTS):
                    if tok in group:
                        out[i, j] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1
        return (out / n).astype("float32")


FILLER = [Doc(f"f{i}", f"unrelated note number {i} about gardening tomatoes and watering schedule {i * 7}") for i in range(12)]
SEMANTIC = Doc("sem", "how to mend a broken vehicle at home")                        # no word overlap with the query
EXACT = Doc("exact", "PostgresSaver stores checkpoints; call setup() once on first run")  # identifier only BM25 knows
DOCS = [*FILLER, SEMANTIC, EXACT]


@pytest.fixture()
def idx():
    return HybridIndex(ConceptEmbedder()).build(DOCS)


def ids(hits):
    return [h.doc.id for h in hits]


# ---------------------------------------------------------------- tokenizer / chunker
def test_tokenize_splits_identifiers_and_cjk():
    t = tokenize("PostgresSaver.get_state 恢复检查点 ok")
    assert {"postgressaver", "postgres", "saver", "get_state", "get", "state", "ok"} <= set(t)
    assert {"恢复", "复检", "检查", "查点"} <= set(t)
    assert tokenize("点") == ["点"]


def test_chunk_text_respects_max_and_overlaps():
    text = "\n\n".join(f"paragraph {i} " + "x" * 300 for i in range(10))
    chunks = chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) > 1 and all(len(c) <= 1000 + 100 + 2 for c in chunks)
    assert "paragraph 0" in chunks[0] and "paragraph 9" in chunks[-1]
    huge = chunk_text("y" * 5000, max_chars=1000, overlap=100)
    assert len(huge) >= 5 and all(len(c) <= 1000 for c in huge)


# ---------------------------------------------------------------- fusion
def test_rrf_prefers_docs_found_by_both_retrievers():
    fused = rrf_fuse({"a": [1, 2, 3], "b": [3, 1, 4]})
    order = [i for i, _ in fused]
    assert order[:2] == [1, 3]                    # in both lists beats single-list docs
    assert dict(fused)[1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_weights_shift_the_outcome():
    fused = rrf_fuse({"a": [1], "b": [2]}, weights={"a": 1.0, "b": 3.0})
    assert fused[0][0] == 2


# ---------------------------------------------------------------- retrieval behaviour
def test_dense_finds_synonym_that_bm25_misses(idx):
    q = "automobile repair"
    assert "sem" not in ids(idx.search(q, k=3, mode="bm25", rerank=False))
    assert ids(idx.search(q, k=3, mode="dense", rerank=False))[0] == "sem"


def test_bm25_finds_identifier_that_dense_misses(idx):
    q = "PostgresSaver setup"
    assert ids(idx.search(q, k=3, mode="bm25", rerank=False))[0] == "exact"


def test_hybrid_recovers_both_kinds_of_query(idx):
    assert "sem" in ids(idx.search("automobile repair", k=3, mode="hybrid", rerank=False))
    assert ids(idx.search("PostgresSaver setup", k=3, mode="hybrid", rerank=False))[0] == "exact"


def test_hits_report_which_retriever_found_them(idx):
    hit = idx.search("PostgresSaver setup", k=1, mode="hybrid", rerank=False)[0]
    assert hit.doc.id == "exact" and "bm25" in hit.ranks and hit.ranks["bm25"] == 1


def test_reranker_reorders_top_candidates():
    class PreferExact:
        name = "prefer-exact"

        def score(self, query, passages):
            return [10.0 if "PostgresSaver" in p else 0.0 for p in passages]

    idx = HybridIndex(ConceptEmbedder(), PreferExact()).build(DOCS)
    q = "automobile repair PostgresSaver"
    assert ids(idx.search(q, k=3, mode="hybrid", rerank=True))[0] == "exact"


def test_unknown_mode_and_duplicate_ids(idx):
    with pytest.raises(ValueError):
        idx.search("x", mode="nope")
    with pytest.raises(ValueError):
        HybridIndex(ConceptEmbedder()).build([Doc("a", "x"), Doc("a", "y")])


def test_chinese_bm25_retrieval():
    docs = [Doc(f"z{i}", f"无关内容第{i}条关于花园和天气") for i in range(10)]
    docs.append(Doc("zh", "如何从检查点恢复一个中断的任务"))
    idx = HybridIndex(HashingEmbedder()).build(docs)
    assert ids(idx.search("恢复中断任务", k=1, mode="bm25", rerank=False)) == ["zh"]


# ---------------------------------------------------------------- persistence
def test_save_load_roundtrip_and_embedder_guard(tmp_path, idx):
    idx.save(tmp_path / "ix")
    loaded = HybridIndex.load(tmp_path / "ix", ConceptEmbedder())
    assert ids(loaded.search("automobile repair", k=3, mode="hybrid", rerank=False)) == ids(idx.search("automobile repair", k=3, mode="hybrid", rerank=False))
    with pytest.raises(ValueError, match="built with"):
        HybridIndex.load(tmp_path / "ix", HashingEmbedder())


# ---------------------------------------------------------------- evaluation
def test_metrics():
    assert recall_at_k(["a", "b", "c"], {"c", "z"}, 2) == 0.0
    assert recall_at_k(["a", "b", "c"], {"c", "z"}, 3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"c", "z"}, 3, strict=True) == 0.5
    assert mrr_at_k(["a", "b", "c"], {"b"}, 5) == 0.5 and mrr_at_k(["a"], {"q"}, 5) == 0.0


def test_evaluate_ablation_shows_hybrid_beats_each_single_retriever():
    queries = [
        {"query": "automobile repair", "relevant": ["sem"]},
        {"query": "PostgresSaver setup", "relevant": ["exact"]},
    ]
    idx = HybridIndex(ConceptEmbedder(), LexicalOverlapReranker()).build(DOCS)
    rep = evaluate(idx, queries, [1, 3])
    r = rep["results"]
    assert r["bm25"]["recall@3"] == 0.5 and r["dense"]["recall@3"] >= 0.5
    assert r["hybrid"]["recall@3"] == 1.0
    assert "hybrid+rerank" in r
    assert "recall@3" in format_report(rep)
