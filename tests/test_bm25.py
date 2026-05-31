"""Tests for BM25Retriever."""

from src.tiny_rag.retrieval.bm25 import BM25Retriever


def test_search_empty_returns_empty():
    retriever = BM25Retriever()
    assert retriever.search("test") == []


def test_add_and_search_finds_relevant():
    retriever = BM25Retriever()
    retriever.add_document(
        doc_id="doc_001", filename="test.txt",
        chunks=[
            "可用性检查功能说明文档",
            "CSS云搜索服务配置指南",
            "OBS对象存储对接手册",
        ],
    )
    results = retriever.search("可用性检查", n_results=2)
    assert len(results) >= 1
    assert results[0]["doc_id"] == "doc_001"
    assert "可用性" in results[0]["text"]


def test_search_ranks_by_keyword_relevance():
    retriever = BM25Retriever()
    retriever.add_document(
        doc_id="doc_001", filename="test.txt",
        chunks=[
            "可用性检查帮助文档可用性检查帮助文档可用性检查帮助文档",
            "权限管理功能说明",
            "可用性检查问题排查指南",
        ],
    )
    results = retriever.search("可用性检查权限", n_results=3)
    # 含"可用性检查"和"权限"的 chunk 应该排前面
    texts = [r["text"] for r in results]
    scores = [r["score"] for r in results]
    assert scores[0] >= scores[-1]  # 递减排序


def test_clear_resets_index():
    retriever = BM25Retriever()
    retriever.add_document("doc_001", "test.txt", ["hello world"])
    assert len(retriever.search("hello")) == 1
    retriever.clear()
    assert retriever.search("hello") == []


def test_multiple_documents_merged():
    retriever = BM25Retriever()
    retriever.add_document("doc_001", "a.txt", ["可用性检查说明"])
    retriever.add_document("doc_002", "b.txt", ["云服务配置"])
    results = retriever.search("可用性检查", n_results=5)
    assert len(results) == 2
