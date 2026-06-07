"""Tests for query_log — SQLite query logging."""

import tempfile
from pathlib import Path

from src.tiny_rag.query_log import QueryLog


def test_log_and_recent():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        log = QueryLog(db_path)
        log.log_query({
            "original_question": "测试问题",
            "rewritten": "测试改写",
            "cache_hit": False,
            "latency_ms": 1234,
            "vector_n": 12,
            "bm25_n": 4,
            "vector_raw": 12,
            "bm25_raw": 3,
            "final_count": 5,
            "src_vector": 4,
            "src_bm25": 0,
            "src_both": 1,
        })
        recent = log.recent(10)
        assert len(recent) == 1
        assert recent[0]["original_question"] == "测试问题"
        assert recent[0]["cache_hit"] == 0
        assert recent[0]["src_vector"] == 4


def test_recent_empty():
    with tempfile.TemporaryDirectory() as tmp:
        log = QueryLog(str(Path(tmp) / "test.db"))
        assert log.recent(10) == []


def test_log_handles_db_error_gracefully():
    """日志写入失败不应抛出异常。"""
    log = QueryLog("/invalid/path/test.db")
    log.log_query({
        "original_question": "t", "rewritten": "t",
        "cache_hit": False, "latency_ms": 0,
        "vector_n": 12, "bm25_n": 4,
        "vector_raw": 0, "bm25_raw": 0, "final_count": 0,
        "src_vector": 0, "src_bm25": 0, "src_both": 0,
    })
    # 不抛异常即为通过
