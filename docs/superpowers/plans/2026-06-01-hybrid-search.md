# Hybrid Search (Vector + BM25) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 ChromaDB 向量检索基础上增加 BM25 关键词检索，通过 RRF 合并两路结果，提升业务专有名词（产品名、功能名）的召回准确率。

**Architecture:** 保持现有 VectorStore 不变，新增 `BM25Retriever` 管理 BM25 倒排索引。上传文档时同时写入两路；检索时各查各的，结果通过 Reciprocal Rank Fusion (RRF) 合并排序。

**Tech Stack:** Python 3.12, rank_bm25 0.2.2, ChromaDB（已有）, numpy（rank_bm25 依赖）

---

### Task 1: 添加 rank_bm25 依赖

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 在 requirements.txt 添加 rank_bm25**

```text
# ============================================================
# Retrieval
# ============================================================
rank_bm25==0.2.2
```

放在 `pymupdf` 之后、`waitress` 之前。

- [ ] **Step 2: 安装依赖**

```bash
pip install -r requirements.txt
```

预期：rank_bm25 0.2.2 安装成功，numpy 自动被依赖安装。

- [ ] **Step 3: 提交**

```bash
git add requirements.txt
git commit -m "chore: add rank_bm25 dependency for BM25 retrieval"
```

---

### Task 2: 创建 BM25Retriever

**Files:**
- Create: `src/tiny_rag/retrieval/__init__.py`
- Create: `src/tiny_rag/retrieval/bm25.py`
- Test: `tests/test_bm25.py`

- [ ] **Step 1: 创建 retrieval 包**

`src/tiny_rag/retrieval/__init__.py` — 空文件。

- [ ] **Step 2: 编写 BM25Retriever 类**

`src/tiny_rag/retrieval/bm25.py`:

```python
"""BM25 keyword retriever — exact-match complement to vector search."""

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 index over document chunks for keyword-style retrieval.

    Maintains an independent copy of chunk texts alongside VectorStore.
    Rebuilds the BM25 index incrementally as documents are added.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def add_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[str],
    ) -> None:
        """Add document chunks to the BM25 index.

        Args:
            doc_id: Unique document identifier.
            filename: Original filename.
            chunks: List of text chunks.
        """
        start = len(self._chunks)
        self._chunks.extend(chunks)
        self._metadatas.extend(
            {"doc_id": doc_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        )

        tokenized_corpus = [self._tokenize(c) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Search by BM25 score.

        Args:
            query: Raw query string.
            n_results: Number of top results.

        Returns:
            List of result dicts with keys: doc_id, filename, chunk_index, text, score.
        """
        if not self._bm25 or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Top N by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:n_results]

        return [
            {
                "doc_id": self._metadatas[i]["doc_id"],
                "filename": self._metadatas[i]["filename"],
                "chunk_index": self._metadatas[i]["chunk_index"],
                "text": self._chunks[i],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]

    def clear(self) -> None:
        """Reset the index (e.g. after document deletion)."""
        self._chunks.clear()
        self._metadatas.clear()
        self._bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace+punctuation tokenizer for Chinese text."""
        return text.lower().split()
```

- [ ] **Step 3: 编写 BM25Retriever 测试**

`tests/test_bm25.py`:

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_bm25.py -v
```

预期：5 passed

- [ ] **Step 5: 提交**

```bash
git add src/tiny_rag/retrieval/ tests/test_bm25.py
git commit -m "feat: add BM25 retriever for keyword-based chunk retrieval"
```

---

### Task 3: 创建 HybridSearch 合并器

**Files:**
- Create: `src/tiny_rag/retrieval/hybrid.py`
- Test: `tests/test_hybrid.py`

- [ ] **Step 1: 编写 RRF 合并函数**

`src/tiny_rag/retrieval/hybrid.py`:

```python
"""Hybrid search — merge vector + BM25 results via RRF."""

_K = 60  # RRF constant


def rrf_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    n_results: int = 5,
) -> list[dict]:
    """Merge two ranked result lists by Reciprocal Rank Fusion.

    Args:
        vector_results: Results from vector search (must have 'text' key).
        bm25_results: Results from BM25 search (must have 'text' key).
        n_results: Number of top results to return.

    Returns:
        Merged and sorted result list (text-deduplicated), keeping all
        keys from the source with higher RRF contribution.
    """
    # Build per-document RRF scores, keyed by chunk text (exact dedup)
    scores: dict[str, float] = {}
    best_result: dict[str, str | float | int] = {}

    for rank, result in enumerate(vector_results, start=1):
        text: str = result["text"]
        scores[text] = scores.get(text, 0.0) + 1.0 / (_K + rank)
        if text not in best_result:
            best_result[text] = result

    for rank, result in enumerate(bm25_results, start=1):
        text = result["text"]
        scores[text] = scores.get(text, 0.0) + 1.0 / (_K + rank)
        if text not in best_result:
            best_result[text] = result

    # Sort by RRF score descending
    ranked = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)[:n_results]

    return [best_result[t] for t in ranked]
```

- [ ] **Step 2: 编写 HybridSearch 测试**

`tests/test_hybrid.py`:

```python
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
```

- [ ] **Step 3: 运行测试确认通过**

```bash
python -m pytest tests/test_hybrid.py -v
```

预期：4 passed

- [ ] **Step 4: 提交**

```bash
git add src/tiny_rag/retrieval/hybrid.py tests/test_hybrid.py
git commit -m "feat: add RRF merge for hybrid vector + BM25 search"
```

---

### Task 4: 集成到 app.py

**Files:**
- Modify: `src/tiny_rag/app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: 在 app.py 中初始化 BM25Retriever（模块级，与 cache 并列）**

当前 `app.py` 的初始化区域（VectorStore、cache、llm 之后）添加：

```python
from src.tiny_rag.retrieval.bm25 import BM25Retriever
from src.tiny_rag.retrieval.hybrid import rrf_merge

# ... 现有初始化 ...

bm25_retriever = BM25Retriever()
```

放置在 `llm = LLMClient(...)` 之后。

- [ ] **Step 2: 修改 upload()，上传文档时同时写入 BM25 索引**

在 `vector_store.add_document(...)` 和 `cache.clear()` 之间插入：

```python
    # 写入 BM25 索引
    bm25_retriever.add_document(doc_id=doc_id, filename=file.filename, chunks=chunks)
```

所以 upload() 的 chunk 处理后部分变为：

```python
    embeddings = embedder.embed(chunks)
    vector_store.add_document(doc_id=doc_id, filename=file.filename, chunks=chunks, embeddings=embeddings)
    bm25_retriever.add_document(doc_id=doc_id, filename=file.filename, chunks=chunks)
    cache.clear()
```

- [ ] **Step 3: 修改 ask()，检索阶段改为 hybrid search**

找到 `ask()` 中的现有检索代码（当前 `vector_store.search(question_embedding, n_results=5)`），替换为：

```python
    # ── 双路检索 + RRF 合并 ──
    vector_results = vector_store.search(question_embedding, n_results=10)
    bm25_results = bm25_retriever.search(question, n_results=10)
    results = rrf_merge(vector_results, bm25_results, n_results=5)
```

注意：保持 `question_embedding` 的生成（`embedder.embed([question])[0]`）不变，保持缓存检查逻辑完全不变，仅替换搜索调用。

完整上下文 — `ask()` 中替换后相关段落：

```python
    question = body["question"]
    force_refresh = body.get("force_refresh", False)

    question_embedding = embedder.embed([question])[0]

    # ── 语义缓存检查（不变）──
    if not force_refresh:
        cached = cache.search(query_embedding=question_embedding)
        ...

    # ── 双路检索 + RRF 合并 ──
    vector_results = vector_store.search(question_embedding, n_results=10)
    bm25_results = bm25_retriever.search(question, n_results=10)
    results = rrf_merge(vector_results, bm25_results, n_results=5)

    if not results:
        return jsonify({"answer": "未找到相关文档，请先上传文档。", "sources": []})
    ...
```

- [ ] **Step 4: 运行现有测试确认集成正确**

```bash
python -m pytest tests/test_app.py tests/test_cache.py tests/test_vector_store.py -v
```

预期：全部通过

- [ ] **Step 5: 提交**

```bash
git add src/tiny_rag/app.py
git commit -m "feat: integrate BM25 hybrid search into upload and ask flow"
```

---

### Task 5: 运行全量测试 + 手动验证

- [ ] **Step 1: 全量测试**

```bash
python -m pytest -v
```

预期：全部测试通过（BM25 5 个 + hybrid 4 个 + 存量测试全部绿）

- [ ] **Step 2: 启动应用**

```bash
python -m src.tiny_rag.app
```

- [ ] **Step 3: 上传带云服务产品名的文档**

```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@data/test.txt"
```

- [ ] **Step 4: 测试业务专有名词检索**

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"可用性检查CSS检查项结果对不上"}'
```

观察返回的 context 中是否包含"可用性检查"和"CSS"相关的内容。

- [ ] **Step 5: 验证缓存仍然正常工作**

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"可用性检查CSS检查项结果对不上"}'
curl http://localhost:5000/stats
```

确认 `hits` 增加。
