"""
QueryLog —— SQLite 日志记录（15 字段，只写不读）

核心认知：
- 每次 /ask 请求（无论缓存命中还是完整检索）都写一条日志
- 15 字段：user_id、original_question、rewritten、latency_ms、
           vector_hits、bm25_hits、final_count、src_vector/src_bm25/src_both、
           user_click
- user_id 是关键：串联点踩反馈和人工群反馈，关联到当天检索日志
- 设计原则：只写不读（不提供查询 API）、写入失败不抛异常只记 warning

API 要点（面试用）：
- sqlite3.connect → 本地文件 data/queries.db
- INSERT INTO query_log (...) VALUES (...)
- 不提供查询 API（不读），外部查询用 sqlite3 命令行直连
"""

import sqlite3
import json
import time
import os

DB_PATH = "data/queries.db"


def init_querylog():
    """建表（应用启动时调一次）"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT    NOT NULL,      -- 企业登录账号，关联人工群反馈的 key
            session_id      TEXT    NOT NULL,      -- 请求会话 ID
            original_question TEXT  NOT NULL,      -- 用户原始问题
            rewritten       TEXT,                  -- 改写后问题
            query_ts        REAL    NOT NULL,      -- 请求时间戳
            latency_ms      REAL,                  -- 端到端延迟
            vector_hits     TEXT,                  -- JSON 数组：向量路召回的 chunk ID 列表
            bm25_hits       TEXT,                  -- JSON 数组：BM25 路召回的 chunk ID 列表
            final_count     INTEGER,               -- Rerank 后 top-5 数量
            src_vector      INTEGER,               -- 最终结果中来自向量路的 chunk 数
            src_bm25        INTEGER,               -- 最终结果中来自 BM25 路的 chunk 数
            src_both        INTEGER,               -- 最终结果中两路重合的 chunk 数
            cache_hit       INTEGER DEFAULT 0,     -- 是否缓存命中（0/1）
            user_click      TEXT,                  -- 用户反馈：赞/踩/空（未点击）
            created_at      TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()
    conn.close()


def write_querylog(user_id: str, session_id: str, original_question: str,
                   rewritten: str, results: list[dict], full_answer: str,
                   latency_ms: float, cache_hit: bool = False):
    """
    写 QueryLog —— 每次 /ask 请求调一次

    设计原则：
    - 写入失败只记 warning，不抛异常 —— 日志不能影响主链路
    - 15 个字段覆盖完整检索过程，排查 bad case 不需要复现
    - user_id 是 join key：人工群反馈 → user_id → QueryLog → 完整检索日志
    """
    # 统计来源分布
    src_vector = sum(1 for r in results if r.get("source") == "vector")
    src_bm25   = sum(1 for r in results if r.get("source") == "bm25")
    src_both   = sum(1 for r in results if r.get("source") == "both")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO query_log (
                user_id, session_id, original_question, rewritten,
                query_ts, latency_ms, vector_hits, bm25_hits,
                final_count, src_vector, src_bm25, src_both, cache_hit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, session_id, original_question, rewritten,
            time.time(), latency_ms,
            json.dumps([r["id"] for r in results if r.get("source") in ("vector", "both")]),
            json.dumps([r["id"] for r in results if r.get("source") in ("bm25", "both")]),
            len(results), src_vector, src_bm25, src_both,
            1 if cache_hit else 0,
        ))
        conn.commit()
    except Exception:
        # 不抛异常，只记 warning —— 日志不能影响主链路
        import logging
        logging.warning("写入 QueryLog 失败", exc_info=True)
    finally:
        conn.close()


def get_user_click(query_id: int) -> str | None:
    """查某次请求的用户反馈（缓存写入时调用，Q23 联动一）"""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT user_click FROM query_log WHERE id = ?", (query_id,)).fetchone()
    conn.close()
    return row[0] if row else None


def update_user_click(query_id: int, click: str):
    """用户点踩/点赞后回写（Q29 路径一）"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE query_log SET user_click = ? WHERE id = ?", (click, query_id))
    conn.commit()
    conn.close()


# --- 人工群反馈回溯（Q29 路径二）---
def find_request_by_user(user_id: str, question_text: str) -> list[dict]:
    """
    人工群反馈 → 根据 user_id 搜索该用户最近的 /ask 记录
    → 用问题文本做相似度匹配 → 找到对应那条日志

    这是两个反馈渠道的 join key：
    聊天工具的 user_id == RAG 系统的 user_id（同一个企业账号）
    """
    conn = sqlite3.connect(DB_PATH)
    # 按 user_id 搜索最近 24h 的记录
    rows = conn.execute("""
        SELECT id, original_question, rewritten, vector_hits, bm25_hits, latency_ms
        FROM query_log
        WHERE user_id = ? AND query_ts > ?
        ORDER BY query_ts DESC
        LIMIT 50
    """, (user_id, time.time() - 86400)).fetchall()
    conn.close()
    # 后续：用 embedding 对 original_question 和 question_text 算语义相似度，
    # 最高分的即是对应那次请求
    return rows
