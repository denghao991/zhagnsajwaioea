# Rerank 精排增强设计

**目标：** 在 BM25+RRF 混合检索之上引入 Cross-encoder Rerank，解决 RRF 仅依赖位置排序、无法理解语义相关性的问题。

**架构：** RRF 合并后的候选集 → DashScope Rerank API 精排 → 取 top5 送 LLM。

**Tech Stack:** DashScope Rerank API（`gte-rerank` 模型），通过 HTTP POST 调用

---

## 数据流

```
用户提问
  │
  ├─→ Embedding → VectorStore.search(n=5)
  ├─→ BM25Retriever.search(n=5)
  │
  └─→ RRF merge (去重，最大 10 条)
       │
       └─→ DashScope Rerank API (re-rank, 取 top5)
            │
            └─→ LLM generate (与现有 prompt 结构一致)
```

### 参数

| 环节 | 取值 | 说明 |
|------|------|------|
| VectorStore.search | n=5 | 向量检索 TOP5 |
| BM25Retriever.search | n=5 | 关键词检索 TOP5 |
| RRF merge | n=10 | 合并去重，最多 10 条 |
| Rerank | top_n=5 | API 精排后取 TOP5 |
| LLM context | 5 chunks | 与现有结构一致 |

### 缓存命中时的行为

语义缓存命中时直接返回缓存结果，**不经过 Rerank**。因为缓存存储的是已经过完整链路（含 rerank）的答案。

---

## 新增模块

### RerankClient（`src/tiny_rag/retrieval/reranker.py`）

```python
class RerankClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None
    def rerank(self, query: str, documents: list[str], top_n: int = 5) -> list[dict]
```

- 调用 DashScope Rerank API：`POST {base_url}/api/v1/services/rerank/text-rerank/text-rerank`
  - 请求体：`{ "model": "{model}", "input": { "query": "...", "documents": [...] }, "parameters": { "top_n": 5 } }`
  - 请求头：`Authorization: Bearer {api_key}`, `Content-Type: application/json`
  - 返回体：`{ "output": { "results": [{ "index": 0, "relevance_score": 0.99 }, ...] }, "usage": {...} }`
- 返回格式与 `rrf_merge` 兼容：`[{text, score, doc_id, filename, chunk_index}, ...]`
- 输入为 RRF 合并后的结果列表，输出为精排后的结果列表
- 使用 `httpx`（openai 的已有依赖）发送 HTTP POST 请求
- 模块级单例，与 embedder/llm 风格一致

### 降级策略

| 场景 | 行为 |
|------|------|
| RRF 结果 ≤ 1 条 | 跳过 rerank，直接使用 |
| RRF 结果 2~5 条 | 仍送 rerank（top_n=实际条数），让模型重新排序 |
| RRF 结果 > 5 条 | 正常 rerank，top_n=5 |
| API 超时/报错 | 捕获异常，回退到 RRF 原始顺序，不影响用户提问 |
| API key 未配置 | 跳过 rerank，直接使用 RRF 结果 |

### Config 新增字段（`src/tiny_rag/config.py`）

```python
rerank_llm_api_key: str = ""
rerank_llm_base_url: str = "https://dashscope.aliyuncs.com"
rerank_llm_model: str = "gte-rerank"
```

对应 `.env` 新增：
```
rerank_llm_api_key=sk-xxxx
rerank_llm_base_url=https://dashscope.aliyuncs.com
rerank_llm_model=gte-rerank
```

---

## 修改点

### `src/tiny_rag/app.py`

- 新增 import: `from src.tiny_rag.retrieval.reranker import RerankClient`
- 模块级初始化: `reranker = RerankClient(...)`
- `/ask` 路由：
  - `vector_store.search` n_results=10 → **5**
  - `bm25_retriever.search` n_results=10 → **5**
  - `rrf_merge` n_results=5 → **10**（保留全集去重结果）
  - 新增 rerank 调用：`reranker.rerank(question, [r["text"] for r in results], top_n=5)`
  - 将 rerank 结果映射回完整 result dict
  - 后续流程（拼 context、SSE 推送、缓存）不变

### 缓存逻辑

缓存命中时跳过 rerank（已缓存的结果已经过 rerank 精排）。缓存未命中或 `force_refresh` 时走完整链路。

---

## 测试

### `tests/test_reranker.py`

- `test_rerank_success` — mock 正常 API 返回，验证排序和 top_n
- `test_rerank_empty_documents` — 空文档列表，返回空
- `test_rerank_api_error` — API 返回错误，优雅降级返回原始顺序
- `test_rerank_integration` — 集成测试（需要真实 API key，conditionally skip）
