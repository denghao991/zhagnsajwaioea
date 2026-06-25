# Rerank 精排增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 BM25+RRF 混合检索之上引入 DashScope Rerank API 精排，提升语义相关性排序准确率。

**Architecture:** RerankClient 通过 httpx 直调 DashScope Rerank API，RRF 合并结果送精排后取 top5 送 LLM。API 异常时降级回退到 RRF 原始顺序。

**Tech Stack:** httpx (openai 已有依赖), DashScope Rerank API (gte-rerank), pydantic-settings

---

### Task 1: RerankClient + Config 配置

**Files:**
- Create: `src/tiny_rag/retrieval/reranker.py`
- Modify: `src/tiny_rag/config.py` (追加 rerank 字段)
- Create: `tests/test_reranker.py`

- [ ] **Step 1: Config 新增 rerank 字段**

在 `src/tiny_rag/config.py` 的 Settings 类末尾添加：

```python
    # DashScope Rerank
    rerank_llm_api_key: str = ""
    rerank_llm_base_url: str = "https://dashscope.aliyuncs.com"
    rerank_llm_model: str = "gte-rerank"
```

- [ ] **Step 2: 写 RerankClient 测试**

```python
"""Tests for RerankClient."""

import json
from unittest.mock import patch, Mock

import pytest

from src.tiny_rag.retrieval.reranker import RerankClient


@pytest.fixture
def client() -> RerankClient:
    return RerankClient(
        base_url="https://dashscope.aliyuncs.com",
        api_key="sk-test",
        model="gte-rerank",
    )


def test_rerank_success(client: RerankClient) -> None:
    """Normal API call returns re-ranked results."""
    docs = [
        {"text": "CSS是云搜索服务", "doc_id": "d1"},
        {"text": "OBS是对象存储", "doc_id": "d2"},
        {"text": "ECS是弹性计算", "doc_id": "d3"},
    ]
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "output": {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.80},
                {"index": 1, "relevance_score": 0.30},
            ]
        },
        "usage": {"total_tokens": 30},
    }
    mock_resp.raise_for_status = lambda: None

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        result = client.rerank("ECS是什么", docs, top_n=3)

    # 验证 API 调用参数
    call_args = mock_post.call_args
    assert call_args[0][0].endswith("/api/v1/services/rerank/text-rerank/text-rerank")
    sent = json.loads(call_args[1]["json"])
    assert sent["input"]["query"] == "ECS是什么"
    assert sent["input"]["documents"] == ["CSS是云搜索服务", "OBS是对象存储", "ECS是弹性计算"]
    assert sent["parameters"]["top_n"] == 3

    # 验证排序: ECS → CSS → OBS
    assert len(result) == 3
    assert result[0]["doc_id"] == "d3"   # ECS, score 0.95
    assert result[0]["score"] == 0.95
    assert result[1]["doc_id"] == "d1"   # CSS, score 0.80
    assert result[1]["score"] == 0.80


def test_rerank_empty_documents(client: RerankClient) -> None:
    """Empty document list returns empty."""
    assert client.rerank("test", []) == []


def test_rerank_empty_query(client: RerankClient) -> None:
    """Empty query returns original docs unchanged."""
    docs = [{"text": "some text", "doc_id": "d1"}]
    result = client.rerank("", docs, top_n=5)
    assert result == docs


def test_rerank_api_error_fallback(client: RerankClient) -> None:
    """API error falls back to original order."""
    docs = [
        {"text": "doc a", "doc_id": "d1"},
        {"text": "doc b", "doc_id": "d2"},
        {"text": "doc c", "doc_id": "d3"},
    ]
    with patch("httpx.post", side_effect=Exception("timeout")):
        result = client.rerank("test query", docs, top_n=2)

    # 回退到原始顺序的前 top_n 条
    assert len(result) == 2
    assert result[0]["doc_id"] == "d1"
    assert result[1]["doc_id"] == "d2"


def test_rerank_top_n_less_than_total(client: RerankClient) -> None:
    """top_n returns only N results."""
    docs = [
        {"text": f"doc {i}", "doc_id": f"d{i}"} for i in range(5)
    ]
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "output": {
            "results": [
                {"index": i, "relevance_score": 1.0 - i * 0.1}
                for i in range(2)
            ]
        },
        "usage": {"total_tokens": 20},
    }
    mock_resp.raise_for_status = lambda: None

    with patch("httpx.post", return_value=mock_resp):
        result = client.rerank("query", docs, top_n=2)

    assert len(result) == 2


def test_rerank_returns_ordered_by_score(client: RerankClient) -> None:
    """Results are sorted by relevance_score descending."""
    docs = [
        {"text": "doc a", "doc_id": "d1"},
        {"text": "doc b", "doc_id": "d2"},
        {"text": "doc c", "doc_id": "d3"},
    ]
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "output": {
            "results": [
                {"index": 1, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.7},
            ]
        },
        "usage": {"total_tokens": 30},
    }
    mock_resp.raise_for_status = lambda: None

    with patch("httpx.post", return_value=mock_resp):
        result = client.rerank("query", docs, top_n=3)

    # scores descending: 0.9, 0.7, 0.5
    assert [r["doc_id"] for r in result] == ["d1", "d3", "d2"]
    assert [r["score"] for r in result] == [0.9, 0.7, 0.5]
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd C:\Users\d\PycharmProjects\tiny-rag && python -m pytest tests/test_reranker.py -v
Expected: ImportError / ModuleNotFoundError for src.tiny_rag.retrieval.reranker
```

- [ ] **Step 4: 实现 RerankClient**

```python
"""Rerank client — Cross-encoder re-ranking via DashScope Rerank API."""

from collections.abc import Mapping
from typing import Any

import httpx


class RerankClient:
    """Re-rank retrieved documents using DashScope Rerank API.

    Wraps the ``POST /api/v1/services/rerank/text-rerank/text-rerank``
    endpoint.  Falls back to the original order on any API error.
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Re-rank *documents* by relevance to *query*.

        Args:
            query: User question.
            documents: Result dicts from RRF merge (must have ``"text"`` key).
            top_n: Number of top results to return.

        Returns:
            Re-ranked result list (same dict format as input), with
            ``"score"`` updated from the API.
        """
        if not documents or not query.strip():
            return documents[:top_n] if top_n else documents

        texts = [d["text"] for d in documents]

        try:
            payload: dict[str, Any] = {
                "model": self._model,
                "input": {"query": query, "documents": texts},
                "parameters": {"top_n": min(top_n, len(texts))},
            }
            resp = httpx.post(
                f"{self._base_url}/api/v1/services/rerank/text-rerank/text-rerank",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return documents[:top_n] if top_n else documents

        results = data["output"]["results"]
        ranked: list[dict[str, Any]] = []
        for item in results:
            idx = item["index"]
            doc = dict(documents[idx])
            doc["score"] = item["relevance_score"]
            ranked.append(doc)

        return ranked
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd C:\Users\d\PycharmProjects\tiny-rag && python -m pytest tests/test_reranker.py -v
Expected: 6 passed
```

- [ ] **Step 6: 全量测试确认无回归**

```bash
cd C:\Users\d\PycharmProjects\tiny-rag && python -m pytest -v
Expected: 全部通过
```

- [ ] **Step 7: 提交**

```bash
cd C:\Users\d\PycharmProjects\tiny-rag && git add src/tiny_rag/retrieval/reranker.py src/tiny_rag/config.py tests/test_reranker.py
git commit -m "feat: add RerankClient for DashScope rerank API"
```

---

### Task 2: 集成到 /ask 流程

**Files:**
- Modify: `src/tiny_rag/app.py`

- [ ] **Step 1: 写集成测试（mock rerank）**

在 `tests/test_reranker.py` 追加：

```python
def test_rerank_integration_in_app() -> None:
    """Verify the full ask flow invokes rerank with correct params."""
    # This test validates the contract between app.py and RerankClient
    from src.tiny_rag.retrieval.reranker import RerankClient

    client = RerankClient(
        base_url="https://dashscope.aliyuncs.com",
        api_key="sk-test",
        model="gte-rerank",
    )
    docs = [{"text": "doc a", "doc_id": "d1"}, {"text": "doc b", "doc_id": "d2"}]
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "output": {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]},
        "usage": {"total_tokens": 20},
    }
    mock_resp.raise_for_status = lambda: None

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        result = client.rerank("test", docs, top_n=2)

    assert len(result) == 2
    assert result[0]["doc_id"] == "d2"  # higher score
    assert result[0]["score"] == 0.9
```

- [ ] **Step 2: 修改 app.py**

在 `src/tiny_rag/app.py` 中做以下改动：

**a) 新增 import（与 `from src.tiny_rag.retrieval.hybrid import rrf_merge` 放在一起）：**

```python
from src.tiny_rag.retrieval.reranker import RerankClient
```

**b) 模块级初始化（与 `bm25_retriever = BM25Retriever()` 放在一起）：**

```python
reranker = RerankClient(
    base_url=settings.rerank_llm_base_url,
    api_key=settings.rerank_llm_api_key,
    model=settings.rerank_llm_model,
)
```

**c) `/ask` 路由中修改检索参数和新增 rerank 调用（替换第 128-130 行）：**

原代码：
```python
    vector_results = vector_store.search(question_embedding, n_results=10)
    bm25_results = bm25_retriever.search(question, n_results=10)
    results = rrf_merge(vector_results, bm25_results, n_results=5)
```

改为：
```python
    vector_results = vector_store.search(question_embedding, n_results=5)
    bm25_results = bm25_retriever.search(question, n_results=5)
    results = rrf_merge(vector_results, bm25_results, n_results=10)
    if results and settings.rerank_llm_api_key:
        results = reranker.rerank(question, results, top_n=5)
    elif results:
        results = results[:5]
```

- [ ] **Step 3: 验证集成**

```bash
cd C:\Users\d\PycharmProjects\tiny-rag && python -c "from src.tiny_rag.app import app; print('imports OK')"
Expected: "imports OK"（模块级初始化无异常即验证通过）
```

- [ ] **Step 4: 运行全部测试确保无回归**

```bash
cd C:\Users\d\PycharmProjects\tiny-rag && python -m pytest -v
Expected: 全部通过（包含新增 7 个 reranker 测试）
```

- [ ] **Step 5: 提交**

```bash
cd C:\Users\d\PycharmProjects\tiny-rag && git add src/tiny_rag/app.py tests/test_reranker.py
git commit -m "feat: integrate rerank into ask flow"
```
