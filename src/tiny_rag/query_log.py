"""QueryLog — SQLite-backed query logging for retrieval analysis."""
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT NOT NULL,
    original_question  TEXT NOT NULL,
    rewritten          TEXT,
    cache_hit          INTEGER DEFAULT 0,
    latency_ms         INTEGER,

    vector_n           INTEGER DEFAULT 12,
    bm25_n             INTEGER DEFAULT 4,

    vector_hits        TEXT DEFAULT '[]',   -- JSON 数组，向量检索 top N 的 chunk_id 列表
    bm25_hits          TEXT DEFAULT '[]',   -- JSON 数组，BM25 检索 top N 的 chunk_id 列表
    final_count        INTEGER DEFAULT 0,

    src_vector         INTEGER DEFAULT 0,
    src_bm25           INTEGER DEFAULT 0,
    src_both           INTEGER DEFAULT 0,

    user_click         TEXT                  -- 前端点击反馈：用户点击的 chunk_id
)
"""


class QueryLog:
    """SQLite 查询日志，用于检索分析。只插入，你自行连库查询。"""

    def __init__(self, db_path: str | None = None) -> None:
        self._path = db_path or str(Path("data/queries.db"))
        self._init_db()

    def _init_db(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(self._path)
            # 迁移：旧表有 vector_raw/bm25_raw → 替换为 vector_hits/bm25_hits
            cursor = conn.execute("PRAGMA table_info(query_log)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            if "vector_raw" in existing_cols:
                conn.execute("DROP TABLE IF EXISTS query_log")
                conn.execute(_SCHEMA)
                logger.info("Migrated query_log schema (old → new)")
            else:
                conn.execute(_SCHEMA)
            conn.commit()
            conn.close()
        except sqlite3.Error:
            logger.warning("Failed to init query log DB", exc_info=True)

    def log_query(self, data: dict) -> None:
        """写入一条查询日志。失败时只记 warning，不抛异常."""
        try:
            conn = sqlite3.connect(self._path)
            conn.execute(
                """INSERT INTO query_log
                   (timestamp, original_question, rewritten, cache_hit, latency_ms,
                    vector_n, bm25_n, vector_hits, bm25_hits, final_count,
                    src_vector, src_bm25, src_both, user_click)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    data.get("original_question", ""),
                    data.get("rewritten"),
                    int(data.get("cache_hit", False)),
                    data.get("latency_ms"),
                    data.get("vector_n", 12),
                    data.get("bm25_n", 4),
                    json.dumps(data.get("vector_hits", []), ensure_ascii=False),
                    json.dumps(data.get("bm25_hits", []), ensure_ascii=False),
                    data.get("final_count", 0),
                    data.get("src_vector", 0),
                    data.get("src_bm25", 0),
                    data.get("src_both", 0),
                    data.get("user_click"),
                ),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error:
            logger.warning("Failed to write query log", exc_info=True)

    def recent(self, n: int = 50) -> list[dict]:
        """返回最近 n 条日志，按时间倒序（仅用于测试和调试）。"""
        try:
            conn = sqlite3.connect(self._path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            logger.warning("Failed to read query log", exc_info=True)
            return []
