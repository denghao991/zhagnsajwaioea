# 缓存精简与配置化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 缓存回归「省 token」的单一职责，去掉中毒/refresh 逻辑，阈值和上限可配置。

**方案：** (1) 去掉双 embedding 和中毒/refresh 整套逻辑；(2) force_refresh 永远创建新条目不覆盖；(3) 缓存阈值和上限搬到 `data/config.yaml`；(4) 超出上限时按创建时间淘汰最旧的。

**涉及文件：**
- 修改：`src/tiny_rag/cache/semantic_cache.py` — 精简语义缓存
- 修改：`src/tiny_rag/app.py` — 去掉双 embedding、中毒判断、refresh 逻辑
- 修改：`src/tiny_rag/config.py` — 加 CACHE_THRESHOLD / CACHE_MAX_ENTRIES
- 修改：`data/config.yaml` — 加 cache 配置段
- 修改：`tests/test_cache.py` — 更新缓存测试
- 修改：`tests/test_app.py` — 更新 app 测试

---

### 任务 1：缓存模块精简

**文件：**
- 修改：`src/tiny_rag/cache/semantic_cache.py`
- 修改：`tests/test_cache.py`

- [ ] **步骤 1：写缓存测试（先改测试）**

新的缓存测试覆盖：搜索空、命中、未命中、clear、max_entries 淘汰。

```python
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
        cache.put(question="q", answer="a", embedding=[0.1, 0.2, 0.3],
                  sources=[], entry_id="e1")
        result = cache.search(query_embedding=[0.1, 0.2, 0.3])
        assert result is not None
        assert result["question"] == "q"
        assert result["answer"] == "a"


def test_put_and_search_miss():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir)
        cache.put(question="q", answer="a", embedding=[0.1, 0.2, 0.3],
                  sources=[], entry_id="e1")
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
        cache.put("q2", "a2", [0.2, 0.3], [], "e2")
        cache.put("q3", "a3", [0.3, 0.4], [], "e3")
        # 此时 3 条，未超上限
        # 再写入一条应触发淘汰
        cache.put("q4", "a4", [0.4, 0.5], [], "e4")
        # e1（最旧）应被淘汰
        # 使用精确匹配阈值确保不会误匹到其他条目
        result = cache.search(query_embedding=[0.1, 0.2])
        assert result is None, "最旧的条目应被淘汰"
        # e4 应当还在
        result = cache.search(query_embedding=[0.4, 0.5])
        assert result is not None and result["question"] == "q4"


def test_search_respects_threshold():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir, threshold=0.05)
        cache.put("q", "a", [0.1, 0.2, 0.3], [], "e1")
        # 距离 0.04（相似 0.96）→ 应在阈值内
        result = cache.search(query_embedding=[0.11, 0.21, 0.30])
        assert result is not None
        # 距离 > 0.05 → 不应命中
        result = cache.search(query_embedding=[0.5, 0.5, 0.5])
        assert result is None
```

- [ ] **步骤 2：运行测试确认失败**

```bash
pytest tests/test_cache.py -v
```
预期：`test_max_entries_evicts_oldest` 和 `test_search_respects_threshold` 因未实现而失败，`test_mark_refreshed_tracks_count` 存在但将被删除。

- [ ] **步骤 3：重写 SemanticCache**

精简后只保留：init、search、put、clear、get_stats，删除 mark_refreshed、get_entry_id、poisoned 相关。

```python
"""Semantic cache — cache LLM responses by question embedding similarity."""

import json
import time
import logging

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """Cache LLM responses keyed by question embedding vectors.

    Uses ChromaDB collection with cosine distance for semantic matching.
    Threshold and max_entries are configurable; oldest entries are evicted
    when the limit is exceeded.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "cache",
        threshold: float = 0.03,
        max_entries: int = 500,
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._threshold = threshold
        self._max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def search(self, query_embedding: list[float]) -> dict | None:
        """Find cached entry by semantic similarity (cosine distance)."""
        if self._collection.count() == 0:
            return None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["metadatas", "distances"],
        )

        if not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        if distance > self._threshold:
            return None

        metadata = results["metadatas"][0][0]
        return {
            "question": metadata.get("question", ""),
            "answer": metadata.get("answer", ""),
            "sources": json.loads(metadata.get("sources", "[]")),
        }

    def put(
        self,
        question: str,
        answer: str,
        embedding: list[float],
        sources: list[dict],
        entry_id: str,
    ) -> None:
        """Store a cache entry, evicting oldest if over max_entries."""
        self._collection.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            metadatas=[{
                "question": question,
                "answer": answer,
                "sources": json.dumps(sources),
                "created_at": time.time(),
            }],
        )
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries when cache exceeds max_entries."""
        count = self._collection.count()
        if count <= self._max_entries:
            return

        all_data = self._collection.get(include=["metadatas"])
        # Sort by created_at ascending (oldest first)
        id_time = []
        for sid, meta in zip(all_data["ids"], all_data["metadatas"]):
            id_time.append((sid, meta.get("created_at", 0)))
        id_time.sort(key=lambda x: x[1])

        excess = count - self._max_entries
        delete_ids = [sid for sid, _ in id_time[:excess + 10]]
        self._collection.delete(ids=delete_ids)

    def clear(self) -> None:
        """Delete all cache entries."""
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

    def get_stats(self) -> dict:
        """Return cache statistics."""
        total = self.hits + self.misses
        return {
            "cache_entries": self._collection.count(),
            "total_requests": total,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total * 100, 1) if total else 0.0,
            "threshold": self._threshold,
            "max_entries": self._max_entries,
        }
```

- [ ] **步骤 4：更新测试文件**

删除 `test_mark_refreshed_tracks_count` 和 `test_get_entry_id_returns_closest`（这两项已被删除）。保留并更新其他测试，移除 `result["poisoned"]` 断言。

- [ ] **步骤 5：运行测试**

```bash
pytest tests/test_cache.py -v
```
预期：所有缓存测试通过（含新加的 max_entries 和 threshold 测试）。

- [ ] **步骤 6：提交**

```bash
git add src/tiny_rag/cache/semantic_cache.py tests/test_cache.py
git commit -m "refactor: simplify cache — remove poisoned/refresh, add max_entries eviction"
```

---

### 任务 2：config.yaml 加缓存配置 + config.py 加载

**文件：**
- 修改：`data/config.yaml`
- 修改：`src/tiny_rag/config.py`

- [ ] **步骤 1：config.yaml 加 cache 段**

```yaml
# 语义缓存
cache:
  threshold: 0.03
  max_entries: 500

# 检索配置
retrieval:
  vector_n: 12
  bm25_n: 4
```

- [ ] **步骤 2：config.py 加 CACHE_THRESHOLD / CACHE_MAX_ENTRIES**

在 `_reload_config()` 中加载 cache 段：

```python
_CACHE_THRESHOLD: float = 0.03
_CACHE_MAX_ENTRIES: int = 500


def _reload_config() -> None:
    global TERM_MAP, VECTOR_N, BM25_N, CACHE_THRESHOLD, CACHE_MAX_ENTRIES
    if not _CONFIG_PATH.exists():
        return
    with open(_CONFIG_PATH, encoding="utf-8") as _f:
        cfg = yaml.safe_load(_f) or {}

    term_map = cfg.get("term_map", {})
    if isinstance(term_map, dict) and term_map:
        TERM_MAP.clear()
        TERM_MAP.update(term_map)

    retrieval = cfg.get("retrieval", {})
    VECTOR_N = retrieval.get("vector_n", 12)
    BM25_N = retrieval.get("bm25_n", 4)

    cache_cfg = cfg.get("cache", {})
    CACHE_THRESHOLD = cache_cfg.get("threshold", 0.03)
    CACHE_MAX_ENTRIES = cache_cfg.get("max_entries", 500)
```

注意添加 `CACHE_THRESHOLD` 和 `CACHE_MAX_ENTRIES` 作为模块级常量（在文件顶部已有 VECTOR_N / BM25_N 的位置一起定义）。

- [ ] **步骤 3：更新 test_config.py 测试**

检查 `test_term_map_values_contain_abbreviation` 不需要改，但应加测试验证 cache 配置被正确加载。追加到 `tests/test_config.py`：

```python
class TestCacheConfig:
    def test_cache_threshold_default(self):
        from src.tiny_rag.config import CACHE_THRESHOLD
        assert isinstance(CACHE_THRESHOLD, float)
        assert CACHE_THRESHOLD > 0

    def test_cache_max_entries_default(self):
        from src.tiny_rag.config import CACHE_MAX_ENTRIES
        assert isinstance(CACHE_MAX_ENTRIES, int)
        assert CACHE_MAX_ENTRIES > 0
```

- [ ] **步骤 4：运行测试**

```bash
pytest tests/test_config.py -v
```
预期：全部通过。

- [ ] **步骤 5：提交**

```bash
git add data/config.yaml src/tiny_rag/config.py tests/test_config.py
git commit -m "feat: add cache config (threshold/max_entries) to config.yaml"
```

---

### 任务 3：去掉双 embedding + 去掉中毒/refresh 逻辑

**文件：**
- 修改：`src/tiny_rag/app.py`
- 修改：`tests/test_app.py`

- [ ] **步骤 1：app.py 去掉双 embedding**

```python
# 改前
_orig_vec, question_vec = embedder.embed([question, rewritten])

# 改后
question_vec = embedder.embed([rewritten])[0]
```

- [ ] **步骤 2：导入 CACHE_THRESHOLD 并传给 SemanticCache 初始化**

在 app.py 中更新初始化：
```python
from src.tiny_rag.config import settings, VECTOR_N, BM25_N, CACHE_THRESHOLD, CACHE_MAX_ENTRIES

cache = SemanticCache(
    persist_dir=settings.chroma_persist_dir,
    threshold=CACHE_THRESHOLD,
    max_entries=CACHE_MAX_ENTRIES,
)
```

- [ ] **步骤 3：去掉中毒判断逻辑**

删除：
```python
if cached and not cached["poisoned"]:   # 改为 if cached:
```
和整个 `elif cached and cached["poisoned"]:` 分支。中毒相关的 `cache.poisoned_skips` 计数器也从缓存初始化中删除。

- [ ] **步骤 4：去掉 force_refresh 的旧条目复用和 mark_refreshed**

当前代码：
```python
if force_refresh:
    cache.force_refreshes += 1
    found_entry_id = cache.get_entry_id(embedding=question_vec)
    entry_id = found_entry_id if found_entry_id else f"cache_{uuid.uuid4().hex[:12]}"
else:
    entry_id = f"cache_{uuid.uuid4().hex[:12]}"
```

改为：
```python
entry_id = f"cache_{uuid.uuid4().hex[:12]}"
```
force_refresh 是否传参不再影响缓存条目的 ID 生成方式。同时删除 `if force_refresh: cache.mark_refreshed(entry_id)` 调用。

注意：`force_refresh` 参数本身在 `app.py` 中保留（用于跳过缓存读），只是去掉它对缓存条目 ID 的影响。

- [ ] **步骤 5：更新 app 测试**

更新 `test_stats_returns_counters` 和 `test_ask_skips_rerank_when_key_empty` 等测试，移除对 `poisoned_skips` / `force_refreshes` 的断言。

```python
def test_stats_returns_counters(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    for key in ("hits", "misses", "total_requests", "hit_rate", "cache_entries", "threshold", "max_entries"):
        assert key in data
```

- [ ] **步骤 6：运行测试**

```bash
pytest tests/test_app.py tests/test_cache.py -v -x
```
预期：全部通过。

- [ ] **步骤 7：提交**

```bash
git add src/tiny_rag/app.py tests/test_app.py
git commit -m "refactor: remove dual embedding, poisoned logic, cache refresh tracking"
```

---

### 任务 4：全量测试与最终确认

- [ ] **步骤 1：运行全量测试**

```bash
python -m pytest -v
```
预期：所有测试通过。

- [ ] **步骤 2：核对需求覆盖**

- [ ] 双 embedding 已去掉，恢复单 embedding
- [ ] 缓存中毒/refresh_count 逻辑已删除
- [ ] force_refresh 不再复用旧条目 ID
- [ ] 缓存阈值从 `data/config.yaml` 读取，默认 0.03
- [ ] 缓存上限从 `data/config.yaml` 读取，默认 500，超出淘汰最旧
- [ ] `cache.get_stats()` 返回 threshold 和 max_entries
- [ ] `data/config.yaml` 包含 cache 配置段
- [ ] 全部测试通过

- [ ] **步骤 3：最终提交**

```bash
git add -A
git commit -m "chore: final cache cleanup and configuration"
```
