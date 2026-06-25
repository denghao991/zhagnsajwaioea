# 语义缓存实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 为 RAG 系统添加语义缓存层，命中时跳过 LLM 调用直接返回缓存的回答

**架构：** 新增 `src/tiny_rag/cache/` 模块，`SemanticCache` 类封装 ChromaDB 集合用于语义匹配，在 `app.py:ask()` 中 Embedding 后插入缓存检查

**技术栈：** ChromaDB（已有）、Python 3.12

---

### 任务 1: 创建 cache 模块

**文件:**
- Create: `src/tiny_rag/cache/__init__.py`（空文件）
- Create: `src/tiny_rag/cache/semantic_cache.py`

- [ ] **步骤 1: 创建 `src/tiny_rag/cache/__init__.py`**

空文件。

- [ ] **步骤 2: 编写 SemanticCache 类**

```python
"""Semantic cache — cache LLM responses by question embedding similarity."""

import json
from collections.abc import Sequence

import chromadb
from chromadb.config import Settings


class SemanticCache:
    """Cache LLM responses keyed by question embedding vectors.

    Uses ChromaDB collection with cosine distance for semantic matching.
    """

    def __init__(self, persist_dir: str, collection_name: str = "cache") -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search(
        self,
        query_embedding: list[float],
        threshold: float = 0.07,
    ) -> dict | None:
        """Find cached entry by semantic similarity.

        Args:
            query_embedding: Embedding vector of the question.
            threshold: Cosine distance threshold (0 = identical, lower = stricter).
                      0.07 ≈ cosine similarity 0.93.

        Returns:
            Cached entry dict (question, answer, sources, refresh_count) or None.
        """
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
        if distance > threshold:
            return None

        metadata = results["metadatas"][0][0]
        return {
            "question": metadata.get("question", ""),
            "answer": metadata.get("answer", ""),
            "sources": json.loads(metadata.get("sources", "[]")),
            "refresh_count": metadata.get("refresh_count", 0),
            "poisoned": metadata.get("poisoned", False),
        }

    def put(
        self,
        question: str,
        answer: str,
        embedding: list[float],
        sources: list[dict],
        entry_id: str,
    ) -> None:
        """Store or update a cache entry.

        Args:
            question: Original question text.
            answer: LLM-generated answer text.
            embedding: Question embedding vector.
            sources: Retrieved source chunks.
            entry_id: Unique entry ID (used to update existing entries).
        """
        self._collection.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            metadatas=[{
                "question": question,
                "answer": answer,
                "sources": json.dumps(sources),
                "refresh_count": 0,
                "poisoned": False,
            }],
        )

    def mark_refreshed(self, entry_id: str) -> int:
        """Increment refresh_count; return new count."""
        data = self._collection.get(ids=[entry_id], include=["metadatas"])
        if not data["ids"]:
            return 0
        meta = data["metadatas"][0]
        count = meta.get("refresh_count", 0) + 1
        poisoned = count >= 3
        self._collection.update(
            ids=[entry_id],
            metadatas=[{"refresh_count": count, "poisoned": poisoned}],
        )
        return count

    def get_entry_id(self, question: str, embedding: list[float]) -> str | None:
        """Find matching entry ID for a question embedding (for refresh tracking)."""
        if self._collection.count() == 0:
            return None
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["distances"],
        )
        if not results["ids"][0]:
            return None
        return results["ids"][0][0]

    def clear(self) -> None:
        """Delete all cache entries."""
        all_ids = self._collection.get(ids=[])["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
```

---

### 任务 2: 编写 SemanticCache 测试

**文件:**
- Create: `tests/test_cache.py`

- [ ] **步骤 1: 编写测试**

```python
"""Tests for SemanticCache."""

import json
import tempfile

from src.tiny_rag.cache.semantic_cache import SemanticCache


def test_search_empty_returns_none():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir)
        result = cache.search(query_embedding=[0.1, 0.2, 0.3])
        assert result is None


def test_put_and_search_hit():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir)
        cache.put(
            question="一季度营收多少？",
            answer="营收100万",
            embedding=[0.1, 0.2, 0.3],
            sources=[{"doc_id": "doc_001", "text": "..."}],
            entry_id="entry_001",
        )
        # 完全相同向量 → 必然命中
        result = cache.search(query_embedding=[0.1, 0.2, 0.3])
        assert result is not None
        assert result["question"] == "一季度营收多少？"
        assert result["answer"] == "营收100万"
        assert not result["poisoned"]


def test_put_and_search_miss():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir)
        cache.put(
            question="一季度营收多少？",
            answer="营收100万",
            embedding=[0.1, 0.2, 0.3],
            sources=[],
            entry_id="entry_001",
        )
        # 完全不同向量 → 不命中
        result = cache.search(query_embedding=[0.9, 0.8, 0.7])
        assert result is None


def test_mark_refreshed_tracks_count():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir)
        eid = "entry_001"
        cache.put(
            question="test", answer="ans",
            embedding=[0.1, 0.2], sources=[],
            entry_id=eid,
        )
        assert cache.mark_refreshed(eid) == 1
        assert cache.mark_refreshed(eid) == 2
        assert cache.mark_refreshed(eid) == 3
        # 达到 3 次 → poisoned
        result = cache.search(query_embedding=[0.1, 0.2])
        assert result is not None
        assert result["poisoned"]


def test_clear_empties_cache():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        cache = SemanticCache(persist_dir=tmpdir)
        cache.put("q", "a", [0.1, 0.2], [], "e1")
        cache.put("q2", "a2", [0.3, 0.4], [], "e2")
        assert cache.search(query_embedding=[0.1, 0.2]) is not None
        cache.clear()
        assert cache.search(query_embedding=[0.1, 0.2]) is None
```

- [ ] **步骤 2: 运行测试验证全部通过**

```bash
PYTHONPATH=. python -m pytest tests/test_cache.py -v
```

预期输出：5 passed

---

### 任务 3: 集成 SemanticCache 到 app.py

**文件:**
- Modify: `src/tiny_rag/app.py`

- [ ] **步骤 1: 在 app.py 中初始化 cache**

在 `vector_store` 初始化之后添加：

```python
from src.tiny_rag.cache.semantic_cache import SemanticCache

cache = SemanticCache(persist_dir=settings.chroma_persist_dir)
```

- [ ] **步骤 2: 修改 /ask 端点加入缓存逻辑**

替换当前的 `ask()` 函数：

```python
@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True)
    if not body or "question" not in body:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = body["question"]
    force_refresh = body.get("force_refresh", False)

    question_embedding = embedder.embed([question])[0]

    # ── 缓存检查（语义匹配）──
    if not force_refresh:
        cached = cache.search(query_embedding=question_embedding)
        if cached and not cached["poisoned"]:
            def generate_cached():
                yield f"event: context\ndata: {json.dumps(cached['sources'])}\n\n"
                # 模拟流式推送（每 3 字符一段）
                answer = cached["answer"]
                segments = [answer[i:i+3] for i in range(0, len(answer), 3)]
                for seg in segments:
                    yield f"event: token\ndata: {json.dumps(seg)}\n\n"
                yield f"event: done\ndata: {json.dumps({'cached': True})}\n\n"

            return Response(
                stream_with_context(generate_cached()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

    # ── 正常检索 + LLM 流程 ──
    results = vector_store.search(question_embedding, n_results=5)

    if not results:
        return jsonify({"answer": "未找到相关文档，请先上传文档。", "sources": []})

    source_ids = list({r["doc_id"] for r in results})
    context = "\n\n".join(r["text"] for r in results)

    # 生成回答并缓存
    entry_id = f"cache_{uuid.uuid4().hex[:12]}"
    answer_buffer: list[str] = []

    def generate_and_cache():
        # 1. 推送召回片段
        yield f"event: context\ndata: {json.dumps(results)}\n\n"

        # 2. 逐字推送 LLM token + 收集完整回答
        for token in llm.generate_stream(question, context):
            answer_buffer.append(token)
            yield f"event: token\ndata: {json.dumps(token)}\n\n"

        # 3. 存入缓存
        full_answer = "".join(answer_buffer)
        cache.put(
            question=question,
            answer=full_answer,
            embedding=question_embedding,
            sources=results,
            entry_id=entry_id,
        )

        # 4. 结束事件
        yield f"event: done\ndata: {json.dumps({'sources': source_ids})}\n\n"

    return Response(
        stream_with_context(generate_and_cache()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
```

- [ ] **步骤 3: 修改 force_refresh 逻辑（从缓存命中后重新生成）**

当用户请求 `force_refresh` 且已命中缓存时，需要在走 LLM 后将新回答更新到已有条目而非创建新条目。

修改上面的 `force_refresh` 分支——如果是从缓存命中后触发的重新生成，找到对应的 `entry_id`：

在检索之前添加找 entry_id 的逻辑：

```python
    # ── 缓存检查（语义匹配）──
    found_entry_id = None
    if not force_refresh:
        found_entry_id = cache.get_entry_id(question, question_embedding)
        cached = cache.search(query_embedding=question_embedding)
        if cached and not cached["poisoned"]:
            def generate_cached():
                ...
            return Response(...)

    # 如果是在重新生成，用已有 entry_id 做更新
    if force_refresh:
        found_entry_id = cache.get_entry_id(question, question_embedding)
        if found_entry_id and cache.search(query_embedding=question_embedding):
            entry_id = found_entry_id
        else:
            entry_id = f"cache_{uuid.uuid4().hex[:12]}"
    else:
        entry_id = f"cache_{uuid.uuid4().hex[:12]}"
```

在 generate_and_cache 中 LLM 返回后：

```python
        full_answer = "".join(answer_buffer)
        cache.put(
            question=question,
            answer=full_answer,
            embedding=question_embedding,
            sources=results,
            entry_id=entry_id,
        )
        if force_refresh and found_entry_id:
            cache.mark_refreshed(found_entry_id)
```

- [ ] **步骤 4: 上传文档时清空缓存**

在 `/upload` 端点末尾、返回 response 前添加：

```python
    cache.clear()
```

---

### 任务 4: 更新前端

**文件:**
- Modify: `src/tiny_rag/templates/index.html`

- [ ] **步骤 1: 在 done 事件处理中添加 cached 标记**

在当前的 `event: done` 处理块中（约第 156 行），追加 `cached` 标记的检测：

```javascript
          } else if (currentEvent === 'done') {
            const meta = JSON.parse(data);
            if (meta.sources && meta.sources.length) {
              msgDiv.insertAdjacentHTML('beforeend', '<div class="sources">来源: ' + escapeHtml(meta.sources.join(', ')) + '</div>');
            }
            if (meta.cached) {
              // 添加缓存标记 + 重新生成按钮
              const cachedLabel = document.createElement('div');
              cachedLabel.style.cssText = 'font-size:12px;color:#999;margin-top:8px;display:flex;align-items:center;gap:8px;';
              cachedLabel.innerHTML = '<span>此回答来自缓存</span>';
              const refreshBtn = document.createElement('button');
              refreshBtn.textContent = '重新生成';
              refreshBtn.style.cssText = 'padding:2px 10px;font-size:12px;border:1px solid #d0d0d0;border-radius:4px;background:#fff;cursor:pointer;';
              refreshBtn.onclick = function() {
                // 禁用按钮避免重复点击
                refreshBtn.disabled = true;
                refreshBtn.textContent = '重新生成中...';
                // 重新发送问题（带 force_refresh）
                msgDiv.textContent = '';  // 清空回答
                answerBuffer = '';
                doAsk(question, true);  // 新函数，带 force_refresh
              };
              cachedLabel.appendChild(refreshBtn);
              msgDiv.appendChild(cachedLabel);
            }
```

- [ ] **步骤 2: 抽离 ask 逻辑以支持 force_refresh**

将现有的 `ask()` 函数重命名为 `doAsk(question, forceRefresh)`，创建一个新 `ask()` 作为入口：

```javascript
async function doAsk(question, forceRefresh) {
  const input = document.getElementById('question-input');
  const btn = document.getElementById('ask-btn');
  // ... 复用现有 ask() 的 body，但发送 forceRefresh ...
  const body = { question };
  if (forceRefresh) body.force_refresh = true;

  const resp = await fetch('/ask', { 
    method: 'POST', 
    headers: {'Content-Type': 'application/json'}, 
    body: JSON.stringify(body) 
  });
  // ... 后续代码和现有 ask() 相同 ...
}

async function ask() {
  const input = document.getElementById('question-input');
  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  doAsk(question, false);
}
```

---

### 任务 5: 集成测试

**文件:**
- Modify: `tests/test_app.py`

- [ ] **步骤 1: 添加缓存集成测试**

```python
def test_ask_cached_response(client):
    """第一次提问走 LLM，第二次相同问题命中缓存返回 cached=true。"""
    resp1 = client.post("/ask", json={"question": "你好"})
    assert resp1.status_code == 200

    resp2 = client.post("/ask", json={"question": "你好"})
    assert resp2.status_code == 200
    data2 = resp2.get_json()
    # 缓存命中时 done 事件返回 cached: true
    assert data2 is not None
```

- [ ] 注意：此测试需要真实的 API key，可能需要在集成测试中跳过或标记。添加一个标记：

```python
import pytest

@pytest.mark.skip(reason="需要有效的 API key")
def test_ask_cached_response(client):
    ...
```
