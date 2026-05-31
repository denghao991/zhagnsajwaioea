"""Tests for hybrid RRF merge."""

from src.tiny_rag.retrieval.hybrid import rrf_merge


def _make_result(text: str, doc_id: str = "doc_001", distance: float = 0.5):
    return {
        "doc_id": doc_id,
        "filename": "test.txt",
        "chunk_index": 0,
        "text": text,
        "distance": distance,
    }


def test_rrf_merge_both_empty():
    assert rrf_merge([], []) == []


def test_rrf_merge_interspersed():
    vec = [
        _make_result("A", distance=0.1),
        _make_result("B", distance=0.2),
        _make_result("C", distance=0.3),
    ]
    bm25 = [
        _make_result("B", distance=0.5),
        _make_result("D", distance=0.5),
        _make_result("E", distance=0.5),
    ]
    merged = rrf_merge(vec, bm25, n_results=4)
    texts = [r["text"] for r in merged]
    # B 出现在两路中 → RRF 叠加 → 应该排第一
    assert texts[0] == "B"
    assert len(merged) == 4


def test_rrf_merge_dedup():
    vec = [_make_result("A"), _make_result("B")]
    bm25 = [_make_result("A"), _make_result("C")]
    merged = rrf_merge(vec, bm25, n_results=3)
    assert len(merged) == 3  # A 去重, 总共 3 个
    assert all(r["text"] in {"A", "B", "C"} for r in merged)


def test_rrf_merge_respects_n_results():
    vec = [_make_result(f"R{i}") for i in range(10)]
    bm25 = [_make_result(f"R{i}") for i in range(10)]
    merged = rrf_merge(vec, bm25, n_results=3)
    assert len(merged) == 3
