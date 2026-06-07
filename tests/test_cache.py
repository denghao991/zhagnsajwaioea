"""Tests for SemanticCache."""

import tempfile

from src.tiny_rag.cache.semantic_cache import SemanticCache


def test_search_empty_returns_none():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir)
        assert cache.search(query_embedding=[0.1, 0.2, 0.3]) is None


def test_put_and_search_hit():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir)
        cache.put(
            question="q",
            answer="a",
            embedding=[0.1, 0.2, 0.3],
            sources=[],
            entry_id="e1",
        )
        result = cache.search(query_embedding=[0.1, 0.2, 0.3])
        assert result is not None
        assert result["question"] == "q"
        assert result["answer"] == "a"


def test_put_and_search_miss():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir)
        cache.put(
            question="q", answer="a",
            embedding=[0.1, 0.2, 0.3], sources=[],
            entry_id="e1",
        )
        assert cache.search(query_embedding=[0.9, 0.8, 0.7]) is None


def test_clear_empties_cache():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir)
        cache.put("q", "a", [0.1, 0.2], [], "e1")
        assert cache.search(query_embedding=[0.1, 0.2]) is not None
        cache.clear()
        assert cache.search(query_embedding=[0.1, 0.2]) is None


def test_max_entries_evicts_oldest():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir, threshold=0.01, max_entries=3)
        cache.put("q1", "a1", [0.1, 0.2], [], "e1")
        cache.put("q2", "a2", [0.3, 0.4], [], "e2")
        cache.put("q3", "a3", [0.5, 0.6], [], "e3")
        cache.put("q4", "a4", [0.7, 0.8], [], "e4")
        # 最旧的 e1 应被淘汰
        assert cache.search(query_embedding=[0.1, 0.2]) is None
        # e4 应该还在
        result = cache.search(query_embedding=[0.7, 0.8])
        assert result is not None
        assert result["question"] == "q4"


def test_search_respects_threshold():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir, threshold=0.05)
        cache.put("q", "a", [0.1, 0.2, 0.3], [], "e1")
        # 距离 < 0.05 → 应命中
        assert cache.search(query_embedding=[0.11, 0.21, 0.30]) is not None
        # 距离 > 0.05 → 不应命中
        assert cache.search(query_embedding=[0.5, 0.5, 0.5]) is None


def test_get_stats_returns_counters():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir)
        cache.put("q", "a", [0.1, 0.2], [], "e1")
        cache.search(query_embedding=[0.1, 0.2])  # hit
        cache.search(query_embedding=[0.9, 0.8])  # miss
        stats = cache.get_stats()
        assert stats["cache_entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0
        assert "threshold" in stats
        assert "max_entries" in stats
