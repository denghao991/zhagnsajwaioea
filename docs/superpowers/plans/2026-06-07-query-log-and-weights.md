# 查询日志与动态权重 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 检索权重改为动态参数、/ask 接入 SQLite 查询日志、SSE 返回改写元数据。

**方案：** (1) 从 `/ask` 请求体读取 `vector_n`/`bm25_n`，不硬编码；(2) 新增 `QueryLog` SQLite 模块，/ask 只负责插入日志（查询你自己连 SQL）；(3) SSE `event: done` 增加 `original_question`/`rewritten`/`cached`。

**涉及文件：**
- 修改：`src/tiny_rag/app.py` — 删除 JSONL 日志函数，接入 QueryLog，改动态权重，加 SSE 元数据
- 创建：`src/tiny_rag/query_log.py` — SQLite 封装（仅插入 + 测试用查询）
- 创建：`tests/test_query_log.py` — 日志模块测试
- 修改：`tests/test_app.py` — 更新 SSE 字段断言

---

### 任务 1：动态权重 + SSE 元数据

**文件：**
- 修改：`src/tiny_rag/app.py`

- [ ] **步骤 1：删除之前加的 JSONL 日志函数**

删除 `_QUERY_LOG_PATH` 常量和 `_log_query()` 函数（之前步骤中已添加的中间代码）。

保留已加的时间导入：
```python
import time
from datetime import datetime, timezone
```

- [ ] **步骤 2：/ask 改为从请求体读检索参数**

当前：
```python
vector_results = vector_store.search(question_embedding, n_results=5)
bm25_results = bm25_retriever.search(question, n_results=5)
```

改为：
```python
vector_n = body.get("vector_n", 12)
bm25_n = body.get("bm25_n", 4)
vector_results = vector_store.search(question_vec, n_results=vector_n)
bm25_results = bm25_retriever.search(question, n_results=bm25_n)
```

同时用 `question_vec` 替代 `question_embedding` 变量名。

- [ ] **步骤 3：SSE done 事件增加元数据**

缓存命中路径：
```python
yield f"event: done\ndata: {json.dumps({'cached': True, 'original_question': question, 'rewritten': rewritten})}\n\n"
```

正常检索路径：
```python
yield f"event: done\ndata: {json.dumps({'sources': source_info, 'cached': False, 'original_question': question, 'rewritten': rewritten})}\n\n"
```

- [ ] **步骤 4：更新测试**

```python
def test_ask_sse_metadata(client, mock_embedding, mock_llm):
    """SSE done 事件包含 original_question/rewritten/cached。"""
    client.post("/upload", data={"file": (io.BytesIO(b"test"), "test.txt")})
    response = client.post("/ask", json={"question": "test question"})
    lines = response.get_data(as_text=True).split("\n")
    done_line = next(l for l in lines if l.startswith("event: done"))
    idx = lines.index(done_line)
    done_data = json.loads(lines[idx + 1].removeprefix("data: "))
    assert done_data.get("original_question") == "test question"
    assert "rewritten" in done_data
    assert "cached" in done_data


def test_ask_with_custom_weights(client, mock_embedding, mock_llm):
    """/ask 请求体传 vector_n/bm25_n 应生效。"""
    # 不需要断言具体检索结果，只验证不报错
    client.post("/upload", data={"file": (io.BytesIO(b"test"), "test.txt")})
    resp = client.post("/ask", json={"question": "test", "vector_n": 8, "bm25_n": 2})
    assert resp.status_code == 200
```

- [ ] **步骤 5：运行测试验证**

```bash
pytest tests/test_app.py -v -x
```

预期：全部通过。

- [ ] **步骤 6：提交**

```bash
git add src/tiny_rag/app.py tests/test_app.py
git commit -m "feat: dynamic vector/BM25 weights, enrich SSE done event"
```

---

### 任务 2：创建 QueryLog 模块

**文件：**
- 创建：`src/tiny_rag/query_log.py`
- 创建：`tests/test_query_log.py`

- [ ] **步骤 1：写测试**

```python
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
    log.log_query({"original_question": "t", "rewritten": "t",
                   "cache_hit": False, "latency_ms": 0,
                   "vector_n": 12, "bm25_n": 4,
                   "vector_raw": 0, "bm25_raw": 0, "final_count": 0,
                   "src_vector": 0, "src_bm25": 0, "src_both": 0})
    # 不抛异常即为通过
```

- [ ] **步骤 2：运行测试确认失败**

```bash
pytest tests/test_query_log.py -v
```

预期：FAIL（模块不存在）。

- [ ] **步骤 3：实现 QueryLog**

```python
"""QueryLog — SQLite-backed query logging for retrieval analysis."""

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

    vector_raw         INTEGER DEFAULT 0,
    bm25_raw           INTEGER DEFAULT 0,
    final_count        INTEGER DEFAULT 0,

    src_vector         INTEGER DEFAULT 0,
    src_bm25           INTEGER DEFAULT 0,
    src_both           INTEGER DEFAULT 0
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
            conn.execute(_SCHEMA)
            conn.commit()
            conn.close()
        except sqlite3.Error:
            logger.warning("Failed to init query log DB", exc_info=True)

    def log_query(self, data: dict) -> None:
        """写入一条查询日志。失败时只记 warning，不抛异常。"""
        try:
            conn = sqlite3.connect(self._path)
            conn.execute(
                """INSERT INTO query_log
                   (timestamp, original_question, rewritten, cache_hit, latency_ms,
                    vector_n, bm25_n, vector_raw, bm25_raw, final_count,
                    src_vector, src_bm25, src_both)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    data.get("original_question", ""),
                    data.get("rewritten"),
                    int(data.get("cache_hit", False)),
                    data.get("latency_ms"),
                    data.get("vector_n", 12),
                    data.get("bm25_n", 4),
                    data.get("vector_raw", 0),
                    data.get("bm25_raw", 0),
                    data.get("final_count", 0),
                    data.get("src_vector", 0),
                    data.get("src_bm25", 0),
                    data.get("src_both", 0),
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
```

- [ ] **步骤 4：运行测试确认通过**

```bash
pytest tests/test_query_log.py -v
```

预期：3 个测试全部 PASS。

- [ ] **步骤 5：提交**

```bash
git add src/tiny_rag/query_log.py tests/test_query_log.py
git commit -m "feat: add SQLite-backed QueryLog"
```

---

### 任务 3：接入 /ask 流程

**文件：**
- 修改：`src/tiny_rag/app.py`

- [ ] **步骤 1：全局初始化 QueryLog**

删除之前加的 JSONL `_log_query` 函数，替换为：

```python
from src.tiny_rag.query_log import QueryLog
query_log = QueryLog()
```

放在 `web_loader = WebLoader(max_depth=20)` 之后。

- [ ] **步骤 2：/ask 增加耗时统计 + 双 embedding**

解析请求体后：
```python
_t0 = time.time()
```

改写后改为批量 embedding：
```python
rewritten = llm.rewrite(question)
_orig_vec, question_vec = embedder.embed([question, rewritten])
```

- [ ] **步骤 3：缓存命中路径写日志**

在缓存命中后、返回前插入：
```python
query_log.log_query({
    "original_question": question,
    "rewritten": rewritten,
    "cache_hit": True,
    "latency_ms": round((time.time() - _t0) * 1000),
    "vector_n": vector_n,
    "bm25_n": bm25_n,
    "vector_raw": 0,
    "bm25_raw": 0,
    "final_count": 0,
    "src_vector": 0,
    "src_bm25": 0,
    "src_both": 0,
})
```

- [ ] **步骤 4：正常检索路径记录来源分布 + 写日志**

在 RRF 合并后（rerank 前）增加来源统计：
```python
vector_texts = {r["text"] for r in vector_results}
bm25_texts = {r["text"] for r in bm25_results}
```

最终结果确定后计算分布：
```python
source_dist = {"vector": 0, "bm25": 0, "both": 0}
for r in results:
    in_v = r["text"] in vector_texts
    in_b = r["text"] in bm25_texts
    if in_v and in_b:
        source_dist["both"] += 1
    elif in_v:
        source_dist["vector"] += 1
    elif in_b:
        source_dist["bm25"] += 1
```

`generate_and_cache` 结束前写日志：
```python
query_log.log_query({
    "original_question": question,
    "rewritten": rewritten,
    "cache_hit": False,
    "latency_ms": round((time.time() - _t0) * 1000),
    "vector_n": vector_n,
    "bm25_n": bm25_n,
    "vector_raw": len(vector_results),
    "bm25_raw": len(bm25_results),
    "final_count": len(results),
    "src_vector": source_dist["vector"],
    "src_bm25": source_dist["bm25"],
    "src_both": source_dist["both"],
})
```

- [ ] **步骤 5：运行测试**

```bash
pytest tests/test_app.py tests/test_query_log.py -v -x
```

预期：全部通过。

- [ ] **步骤 6：提交**

```bash
git add src/tiny_rag/app.py
git commit -m "feat: integrate QueryLog into /ask flow"
```

---

### 任务 4：全量测试与最终确认

- [ ] **步骤 1：运行全量测试**

```bash
pytest --ignore-glob='test_app.py' -v
```

预期：80+ 测试通过。

- [ ] **步骤 2：运行集成测试**

```bash
pytest tests/test_app.py -v
```

预期：全部通过。

- [ ] **步骤 3：核对需求覆盖**

- [ ] 检索权重改为请求体传参（`vector_n`/`bm25_n`），默认 12/4
- [ ] SSE done 事件包含 `original_question`、`rewritten`、`cached`
- [ ] QueryLog SQLite 模块已创建，仅插入
- [ ] 缓存命中与未命中路径均已接入日志
- [ ] 数据库生成在 `data/queries.db`
- [ ] 全部测试通过

- [ ] **步骤 4：最终提交**

```bash
git add -A
git commit -m "chore: final integration and cleanup"
```
