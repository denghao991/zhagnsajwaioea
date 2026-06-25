# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

本文件为 Claude Code 在此仓库中工作提供指导。

## 项目上下文

开始编码前，必须阅读以下文档
- README.md
- requirements.txt

所有方案严格遵守这些文档完成。

### 环境变量（.env）

项目根目录的 `.env` 文件驱动所有外部服务配置：

| 变量名 | 必需 | 默认值 | 说明 |
|--------|------|--------|------|
| `llm_api_key` | 是 | — | GLM API Key |
| `llm_base_url` | 否 | `https://open.bigmodel.cn/api/paas/v4` | GLM API 地址 |
| `llm_model` | 否 | `glm-4.7` | GLM 模型名 |
| `dashscope_api_key` | 是 | — | 阿里云 DashScope API Key（Embedding） |
| `dashscope_base_url` | 否 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Embedding API 地址 |
| `embedding_model` | 否 | `text-embedding-v2` | Embedding 模型名 |
| `rerank_llm_api_key` | 否 | — | DashScope Rerank API Key（为空时跳过重排） |
| `rerank_llm_base_url` | 否 | `https://dashscope.aliyuncs.com` | Rerank API 地址 |
| `rerank_llm_model` | 否 | `gte-rerank` | Rerank 模型名 |
| `chroma_persist_dir` | 否 | `./chroma_db` | ChromaDB 持久化目录 |
| `chunk_size` | 否 | `512` | 分块 token 数 |
| `chunk_overlap` | 否 | `64` | 分块重叠 token 数 |

## 技术栈

- **Python 3.12**，无 pyproject.toml（使用 pip + requirements.txt）
- **Flask** — 轻量 Web 框架，同步
- **ChromaDB** — 本地向量数据库（`chromadb.PersistentClient`），用于向量存储和语义缓存
- **OpenAI SDK** — 统一调用 LLM（GLM）和 Embedding（Qwen text-embedding-v2）API
- **tiktoken** — cl100k_base 编码，用于固定 token 数分块
- **PyMuPDF (fitz)** — PDF 文本提取（已不再使用，保留依赖防回归）
- **pydantic-settings** — `.env` 配置管理
- **waitress** — 生产级 WSGI 服务器（支持 SSE 流式推送）
- **jieba + rank_bm25** — 中文分词和 BM25 关键词检索
- **httpx** — Rerank API 调用（DashScope 非 OpenAI 兼容端点）
- **pytest + pytest-cov** — 测试框架

## 数据流

```
上传: file → load_bytes → MarkdownChunker.chunk_text → embed → VectorStore.add_document
      + BM25Retriever.add_document → SemanticCache.clear

提问: question → embed → SemanticCache.search (命中→直接返回)
      miss → VectorStore.search(向量) + BM25Retriever.search(关键词)
           → rrf_merge(RRF合并) → RerankClient.rerank(可选)
           → LLMClient.generate_stream → SSE推送 → SemanticCache.put
```

## 核心模块与数据流

- `app.py` — Flask 入口，路由：`/`、`/upload`、`/ask`（SSE 流式）、`/documents`、`/stats`
- `config.py` — `Settings` dataclass（pydantic-settings），从 `.env` 读取两种 API 密钥体系：
  - LLM（GLM，OpenAI 兼容接口）→ `llm_api_key`
  - Embedding + Rerank（DashScope）→ `dashscope_api_key` / `rerank_llm_api_key`
- `ingestion/loader.py` — `load_text()`、`load_bytes()`
- `ingestion/tokenizer.py` — `count_tokens()` / `encode()` / `decode()`，tiktoken cl100k_base 封装
- `ingestion/chunker.py` — `MarkdownChunker` 类：基于 LangChain `MarkdownHeaderTextSplitter` 按标题切分（仅 `#` / `##`）+
  `RecursiveCharacterTextSplitter` 段落累积分块
- `embedding/client.py` — `EmbeddingClient.embed(texts)` 批量生成向量
- `storage/vector_store.py` — `VectorStore` 封装 ChromaDB：`add_document`、`search`（余弦距离）、`list_documents`
- `retrieval/bm25.py` — `BM25Retriever`，jieba 分词 + BM25Okapi 关键词检索（上传时构建，每次 add 全量重建）
- `retrieval/hybrid.py` — `rrf_merge(vector_results, bm25_results, n_results)`，RRF（k=60）合并双路检索结果，按 text 去重
- `retrieval/reranker.py` — `RerankClient`，通过 DashScope Rerank API（httpx 直调）对 RRF 结果做 cross-encoder 重排；API 失败时降级为原顺序
- `cache/semantic_cache.py` — `SemanticCache`，独立 ChromaDB collection 按 embedding 余弦距离缓存 LLM 回答；阈值 0.03（可配置），超出 max_entries 时淘汰最旧条目；force_refresh 参数可跳过缓存但无追踪计数
- `generation/llm.py` — `LLMClient.generate()`（同步）和 `generate_stream()`（流式，用于 SSE）

### SSE 推送协议（`/ask` 端点）

`/ask` 返回 `text/event-stream`，包含三种事件：
1. `event: context` — JSON 串，包含召回的文档片段列表（text、doc_id、score）
2. `event: token` — 每个 LLM token 单独推送
3. `event: done` — 结束事件，附带 `{sources: [...]}`（缓存命中时附带 `{cached: true}`）

`/ask` 支持 `force_refresh` 参数跳过缓存直接请求 LLM。

### 关键设计决策

- 所有第三方 API（LLM + Embedding）通过 OpenAI 兼容 SDK 调用；Rerank 使用 DashScope 原生 API（httpx 直调）
- ChromaDB 使用 PersistentClient 本地持久化，路径由 `.env` 的 `chroma_persist_dir` 控制（默认 `./chroma_db`）
- 文件上传仅支持 `.md` 文件
- embedding 和 LLM 客户端在模块级别初始化（模块级单例）
- BM25 索引在每次 `add_document` 时全量重建（数据量小，设计简单）
- 语义缓存在每次上传新文档后 `clear()`，保证数据一致性

## 常用命令

```bash
# 运行应用（开发）
python -m src.tiny_rag.app

# 运行全部测试
pytest

# 带覆盖率
pytest --cov=src.tiny_rag

# 运行单个测试文件
pytest tests/test_chunker.py

# 运行单个测试函数（-k 模糊匹配）
pytest tests/test_chunker.py -k "test_markdown_single"

# 格式化
black src/ tests/

# 安装依赖
pip install -r requirements.txt
```

## 测试策略

- **纯函数**（tokenizer、chunker、loader）用真实数据测试，无需 API key
- **外部 API 调用**（embedding、LLM、reranker）用 `unittest.mock.patch` mock，无需 API key
- **`vector_store` 和 `semantic_cache`** 用 `tempfile.TemporaryDirectory` 隔离 ChromaDB 持久化目录
- **`app` 集成测试**：大部分路由测试（/documents、/stats、/ask 参数校验）mock 了 API 调用可独立运行；`test_markdown_upload` 需要有效 `.env` 才能通过
- **BM25 测试**使用中文样本验证 jieba 分词 + 排名正确性
- **hybrid 测试**验证 RRF 合并逻辑和 text 去重行为

```bash
# 运行所有测试（含需要 API key 的）
pytest

# 仅运行无需 API key 的测试（过滤掉标记为需要 .env 的）
pytest --ignore-glob='test_app.py' -k 'not test_markdown_upload'

# 仅运行纯函数模块
pytest tests/test_tokenizer.py tests/test_chunker.py tests/test_loader.py
```

## 项目目录结构

```
tiny-rag/
├── src/tiny_rag/
│   ├── app.py                 # Flask 入口 + 路由
│   ├── config.py              # pydantic-settings 配置
│   ├── ingestion/
│   │   ├── loader.py          # 文档加载（txt/md/pdf）
│   │   ├── chunker.py         # 固定 token 分块
│   │   └── tokenizer.py       # tiktoken 封装
│   ├── embedding/
│   │   └── client.py          # Qwen Embedding 客户端
│   ├── storage/
│   │   └── vector_store.py    # ChromaDB 封装
│   ├── retrieval/
│   │   ├── bm25.py            # BM25 关键词检索（jieba 分词）
│   │   ├── hybrid.py          # RRF 合并（向量 + BM25）
│   │   └── reranker.py        # DashScope Rerank API 重排
│   ├── cache/
│   │   └── semantic_cache.py  # 语义缓存（ChromaDB）
│   ├── generation/
│   │   └── llm.py             # GLM 客户端（同步 + 流式）
│   └── templates/
│       └── index.html         # 单页 Web 界面
├── tests/                     # pytest 测试
├── scripts/                   # 分析/审计工具（hybrid_compare.py, token_audit.py）
├── data/                      # 测试文档
├── chroma_db/                 # ChromaDB 持久化目录（gitignore 中）
├── docs/superpowers/          # 设计文档
├── .env                       # API 密钥（不提交）
└── requirements.txt
```
