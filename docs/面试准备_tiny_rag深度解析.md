# Tiny RAG — 面试准备深度解析

> 本文档涵盖：业务流全景、模块设计细节、选型决策对比、关键设计权衡、面试问答集锦。
> 目标：让你在面试中能流畅深入地讨论这个项目的每个细节。

---

## 目录

1. [项目总览](#1-项目总览)
2. [完整业务流（逐步骤详解）](#2-完整业务流逐步骤详解)
3. [模块深度拆解](#3-模块深度拆解)
   - 3.1 文档加载层
   - 3.2 文本分块（Chunking）
   - 3.3 Embedding
   - 3.4 向量存储（ChromaDB）
   - 3.5 双路检索（向量 + BM25）
   - 3.6 RRF 混合融合
   - 3.7 Rerank 重排
   - 3.8 语义缓存
   - 3.9 查询改写
   - 3.10 LLM 生成
   - 3.11 Web 爬虫
   - 3.12 查询日志
4. [选型决策全景](#4-选型决策全景)
5. [关键设计决策与权衡](#5-关键设计决策与权衡)
6. [评估管线与量化分析](#6-评估管线与量化分析)
7. [面试问答集锦](#7-面试问答集锦)

---

## 1. 项目总览

### 一句话概括

Tiny RAG 是一个**从零构建的轻量级检索增强生成系统**，支持文档上传、Web 页面抓取、混合检索（稠密向量 + 稀疏关键词）、Cross-Encoder 重排、语义缓存、查询改写，通过 SSE 流式推送回答。

### 核心数据流

```
                    ┌─────────────┐
                    │  用户上传文档 │
                    └──────┬──────┘
                           ▼
              ┌──────────────────────┐
              │  loader: txt/md/pdf  │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  MarkdownChunker     │  ← 按标题切分 + 按 token 数分块
              │  (chunk_size=512)    │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  EmbeddingClient     │  ← Qwen text-embedding-v2
              │  (OpenAI 兼容接口)    │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐     ┌──────────────────────┐
              │  VectorStore          │     │  BM25Retriever        │
              │  (ChromaDB 向量库)     │     │  (jieba + BM25Okapi)  │
              └──────────┬───────────┘     └──────────┬───────────┘
                         │                           │
                         ▼                           ▼
              ┌─────────────────────────────────────────┐
              │       rrf_merge (RRF 混合融合)            │
              │       alpha=3.0 (向量), beta=10.0 (BM25)   │
              └──────────────────┬──────────────────────┘
                                 ▼
              ┌─────────────────────────────────────────┐
              │  RerankClient (DashScope gte-rerank)     │
              │  (Cross-Encoder 重排 top5)                │
              └──────────────────┬──────────────────────┘
                                 ▼
              ┌─────────────────────────────────────────┐
              │  LLMClient.generate_stream (GLM 4.7)     │
              │  SSE 流式推送 token                       │
              └──────────────────┬──────────────────────┘
                                 ▼
              ┌─────────────────────────────────────────┐
              │  SemanticCache.put (写入缓存)              │
              │  query_log.log_query (记录日志)           │
              └─────────────────────────────────────────┘
```

---

## 2. 完整业务流（逐步骤详解）

### 2.1 文档上传（`POST /upload`）

```
用户选择文件 → 前端 FormData → /upload 路由
```

**步骤拆解：**

1. **文件接收** — Flask 从 `request.files` 获取上传文件
2. **格式校验** — 检查后缀名是否在 `ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}`
3. **内容提取**：
   - `.pdf` → `load_pdf()` 使用 PyMuPDF（fitz）逐页提取文本
   - `.txt` / `.md` → `load_bytes()` 直接 UTF-8 解码
4. **生成 doc_id** — `doc_uuid_hex[:12]` 格式，12 位十六进制，保证唯一
5. **分块** — `MarkdownChunker.chunk_text(content)`，返回 `list[ChunkResult]`
6. **向量化** — `EmbeddingClient.embed([chunk.text for chunk in chunks])`，批量请求（不是逐条）
7. **双路存储**：
   - VectorStore → ChromaDB，存 embedding + 元数据
   - BM25Retriever → 内存中维护文本列表 + jieba 分词后构建 BM25Okapi 索引
8. **缓存失效** — 新文档上传意味着知识库变化，`SemanticCache.clear()` 清空所有缓存
9. **返回** — `{id, filename, chunks}`（分块数量）

**注意点：** BM25 索引是「全量重建」的——每次 `add_document` 将新 chunks 追加到 `_chunks` 列表后，对整个列表重新 `jieba.lcut` 分词并构建 `BM25Okapi`。数据量小（几百个 chunk）时无性能问题，量大时需要增量索引优化。

### 2.2 Web 页面上传（`POST /upload_web`）

```
用户输入 URL → BFS 爬取 → 每页独立转 Markdown → 同文档上传流程
```

**步骤拆解：**
1. 接收 `{"url": "...", "max_depth": 20}`
2. `WebLoader.load(url, max_depth)` 启动 BFS 遍历
3. 使用 `html2text` 将 HTML 转 Markdown
4. **关键设计**：去掉 Markdown 中的图片引用 `![...](...)`，避免噪声
5. 每页作为一个独立 `doc_id` 走完整的分块→向量化→双路存储流程
6. 同样触发 `cache.clear()`

### 2.3 提问回答（`POST /ask`）— **核心链路**

```
用户输入问题 → SSE 流式返回
```

这是最复杂的流程，共 8 个阶段：

#### 阶段 1：查询改写

```python
rewritten = llm.rewrite(question)
```

- 调用 LLM（非规则方法）将口语化问题改写为规范的文档术语表述
- 使用 few-shot 示例（`REWRITE_EXAMPLES`）+ 术语映射（`TERM_MAP`）+ 推理规则
- 例如 `"CSS可用区未多AZ这个是啥意思"` → `"云服务CSS可用区未多AZ，这个风险检查项是什么意思？"`
- **为什么用 LLM 而不是规则？** 规则只能做缩写替换，但检查项名称推理需要语义理解——LLM 可以理解"CSS 可用区未多 AZ"是一个风险检查项名称

#### 阶段 2：向量化改写后的问题

```python
question_vec = embedder.embed([rewritten])[0]
```

#### 阶段 3：语义缓存查询

```python
if not force_refresh:
    cached = cache.search(query_embedding=question_vec)
```

- 用改写后的问题 embedding 在缓存 ChromaDB collection 中做余弦距离搜索
- 阈值 `threshold=0.03`（余弦距离），**低于**此值视为语义相同→命中缓存
- 命中缓存时：直接 SSE 推送缓存的回答。注意缓存路径和非缓存路径的 SSE 格式有细微差异：
  - 缓存路径的 `token` 事件推送的是 3 字符切片（而非逐 token），减少事件数量
  - 缓存路径的 `done` 事件**不携带** `sources` 字段（来源已在 context 事件中推送）
  - 前端会显示"此回答来自缓存"+"重新生成"按钮，点击后以 `force_refresh=true` 重试

#### 阶段 4：双路检索

```python
vector_results = vector_store.search(question_vec, n_results=VECTOR_N)   # VECTOR_N=12
bm25_results = bm25_retriever.search(question, n_results=BM25_N)          # BM25_N=4
```

**向量路：**
- 使用改写后的问题 embedding
- 在 ChromaDB 中用欧氏距离（L2，ChromaDB 默认度量）搜索
- 取 top 12（`VECTOR_N=12`）
- 返回：`doc_id, filename, chunk_index, heading_path, text, distance`

**BM25 路：**
- 使用**原始问题**（而非改写版）——关键词匹配角度，原问题更保留用户用词习惯
- jieba 分词后计算 BM25 分数
- 取 top 4（`BM25_N=4`）
- 为什么 BM25 只取 4 条？BM25 召回稀疏，取多了噪声大；向量取 12 条，因为稠密检索通常召回更多语义近似的候选

#### 阶段 5：RRF 混合融合

```python
results = rrf_merge(vector_results, bm25_results, n_results=10,
                     alpha=VECTOR_ALPHA, beta=BM25_BETA)
```

- RRF（Reciprocal Rank Fusion）公式：`score(t) = Σ α / (K + rank_v(t)) + Σ β / (K + rank_b(t))`
- `K=60`（RRF 常数），`alpha=3.0`（向量权重），`beta=10.0`（BM25 权重）
- 权重设计的逻辑：
  - 向量结果 12 条 × alpha=3 = 基础总分 36（12 条各贡献约 3/K ~ 3/63）
  - BM25 结果 4 条 × beta=10 = 基础总分 40（4 条各贡献约 10/K ~ 10/63）
  - beta > alpha 是因为 BM25 条目少但精确度更高，需要单条贡献更大
- 按 text 内容去重（同一个 chunk 可能同时被两路召回）
- 取 top 10

#### 阶段 6：Rerank 重排（可选）

```python
if results and settings.rerank_llm_api_key:
    results = reranker.rerank(question, results, top_n=5)
elif results:
    results = results[:5]
```

- 如果有配置 Rerank API key，用 DashScope gte-rerank 模型对 top 10 做 cross-encoder 重排，取 top 5
- 没有配置则直接截取 top 5
- Rerank API 失败时降级为原顺序（`try-except` 兜底）

#### 阶段 7：构建 LLM 上下文

```python
context = "\n\n".join(r["text"] for r in results)
```

- 将 top 5 检索结果的文本用双换行拼接
- 同时记录来源分布（`source_dist`）用于日志分析

#### 阶段 8：流式生成 + 缓存 + 日志

```python
for token in llm.generate_stream(rewritten, context):
    yield f"event: token\ndata: {json.dumps(token)}\n\n"
```

- 使用 SSE（Server-Sent Events）协议推送三种事件：
  - `event: context` — 召回的文档片段列表（用于前端展示"找到以下相关内容"）
  - `event: token` — 每个 LLM token（逐个推送，前端逐个追加）
  - `event: done` — 结束事件，附带来源信息和缓存标记
- 回答收集完成后：
  1. `cache.put(...)` — 写入语义缓存供下次命中
  2. `query_log.log_query(...)` — 记录完整查询日志（原问题、改写后、延迟、来源分布等）

### 2.4 其他路由

- `GET /documents` — 从 ChromaDB 元数据汇总文档列表（含每个文档的 chunk 数）
- `GET /stats` — 缓存统计（条目数、命中率、阈值等）

---

## 3. 模块深度拆解

### 3.1 文档加载层（`ingestion/loader.py`）

| 方法 | 输入 | 实现 |
|------|------|------|
| `load_text(path)` | 文件路径 | `Path.read_text(encoding="utf-8")` |
| `load_bytes(content)` | bytes | `content.decode("utf-8")` |
| `load_pdf(content)` | PDF bytes | PyMuPDF (fitz) 逐页提取 |

**设计意图：** 统一输出为纯文本字符串，上层（chunker）不需要关心原始格式。

**为什么用 PyMuPDF？** 它是 Python 生态中 PDF 文本提取速度最快的库之一，纯 C 实现，比 pdfminer 快 10x+。不需要处理复杂布局（RAG 场景只关心文本内容），也不需要 OCR。

### 3.2 文本分块（`ingestion/chunker.py`）

**两阶段分块策略：**

```
原始文本
    │
    ▼
第一阶段：MarkdownHeaderTextSplitter（按 # / ## 标题切分）
    │  产出：语义段落（保留标题层级）
    ▼
第二阶段：RecursiveCharacterTextSplitter（用 token 计数）
    │  产出：固定 token 大小的块
    ▼
后处理：在块前拼接标题路径（如 "产品介绍 > 核心功能 > ……"）
```

**关键参数：**
- `chunk_size=512` — 经过评估选型确定（见第 6 章）
- `chunk_overlap=64` — 保证段落边界不丢失上下文
- `length_function=count_tokens` — 使用 tiktoken cl100k_base 编码

**为什么用 tiktoken 而非 len()？**
不同语言的 token 密度不同，用字符数分块会导致实际 token 数波动很大。tiktoken 能精确按 LLM 实际的 token 边界分块，保证每个 chunk 不超过模型上下文限制。

**为什么用两阶段？**
- 只按 token 数硬切分 → 打断语义段落
- 只按标题切分 → 大章节可能远超 token 限制
- 两阶段结合：先保语义边界，再在段内按 token 数调整

**"标题路径"的作用：**
分块时在块内容前拼接 `h1 > h2 > …` 的标题路径，相当于给每个块注入了章节上下文。LLM 生成时可以感知"这段内容来自文档的哪个章节"。

### 3.3 Embedding（`embedding/client.py`）

```python
class EmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]
```

**选型：Qwen text-embedding-v2**
- 通过 **OpenAI 兼容接口**（`/compatible-mode/v1`）调用 DashScope API
- 维度：未显式指定，text-embedding-v2 为 1536 维
- 优势：中文 embedding 效果经过阿里云优化，免费额度充裕

**为什么用 OpenAI SDK 而非直接 HTTP？**
DashScope 提供了 OpenAI 兼容接口，所以可以直接用标准的 `openai` Python SDK。好处是代码简洁、重试逻辑 SDK 内置、接口一致。Rerank 则不同（见下文），因为 DashScope 的 Rerank API 不是 OpenAI 兼容格式，需要用 httpx 直调。

### 3.4 向量存储（`storage/vector_store.py`）

**封装 ChromaDB PersistentClient：**

| 方法 | 功能 |
|------|------|
| `add_document(doc_id, filename, chunks, embeddings)` | 批量写入 |
| `search(query_embedding, n_results)` | L2 欧氏距离搜索（ChromaDB 默认）|
| `list_documents()` | 统计每个文档的 chunk 数 |

**为什么用 ChromaDB 而非 FAISS / Milvus / Pinecone？**

| 对比项 | ChromaDB | FAISS | Milvus | Pinecone |
|--------|----------|-------|--------|----------|
| 部署 | 嵌入式（零依赖） | 需自己管理索引 | 需 Docker 部署 | 云服务 |
| 持久化 | PersistentClient 自带 | 需自己实现 | 自带 | 托管 |
| 元数据过滤 | 内置支持 | 无 | 支持 | 支持 |
| 学习成本 | 低（pip install 即用） | 中 | 高 | 低 |

关键决策：项目定位是**学习用轻量系统**，ChromaDB 的 `pip install` 即用、零运维、自动持久化的特性最适合。FAISS 虽然更快但缺乏元数据管理，需要额外写序列化逻辑。

### 3.5 双路检索

#### 稠密检索（向量）

```
问题文本 → Embedding → 1536 维向量 → ChromaDB L2 欧氏距离搜索 → top 12
```

优点：语义匹配，能处理同义词、不同表述。

缺点：对**精确术语、缩写、编号**不敏感。"CCE集群版本升级"这种包含精准产品名的查询，向量可能把"CCE"分散到语义空间的不同区域。

#### 稀疏检索（BM25）

```
问题文本 → jieba 分词 → BM25Okapi 词频-逆文档频率评分 → top 4
```

BM25 评分公式（简化）：`score(D,Q) = Σ IDF(q_i) × TF(q_i, D) × (k₁ + 1) / (TF + k₁ × (1 - b + b × |D|/avgdl))`

- `k₁=1.5`, `b=0.75`（BM25Okapi 默认）
- jieba 分词：`"CSS可用区未多AZ"` → `["CSS", "可用区", "未", "多", "AZ"]`

优点：精确词匹配，对专业术语、代码、编号非常有效。

缺点：语义泛化能力差，同义词不通配。

#### 为什么用双路检索？

单一稠密检索在专业领域（技术文档、医疗、法律）中，对精确术语的召回率可能不足。双路结合可以互补：
- 向量路兜底语义泛化
- BM25 路提供精确命中

### 3.6 RRF 混合融合（`retrieval/hybrid.py`）

```
RRF 公式：score(text) = alpha/(K + rank_v) + beta/(K + rank_b)

其中：
- K = 60（RRF 常数，防止分母过小）
- alpha = 3.0（向量权重）
- beta = 10.0（BM25 权重）
- rank_v = 该文本在向量结果中的排名（1-based）
- rank_b = 该文本在 BM25 结果中的排名
```

**核心设计：**
- 按 `text` 字段去重（兼顾两路召回同一 chunk 的情况）
- 去重时**优先保留向量路的结果字典**（因为向量路的结果包含 distance 字段）
- 最终按 RRF 分数降序排列，取 top N

**权重的演化历史（重要面试点）：**

| 版本 | alpha | beta | 问题 | 解决 |
|------|-------|------|------|------|
| 初始 | 7.0 | 3.0 | BM25 结果永远进不了 top 10 | 向量 12×7=84 分 vs BM25 4×3=12 分 |
| 调整后 | 3.0 | 10.0 | 需要 BM25 精确匹配有竞争力 | 向量 12×3=36 分 vs BM25 4×10=40 分 |

**为什么 K=60？**
RRF 论文（Cormack et al. 2009）建议 K=60 为经验最优值。K 越大，排名之间的分差越小，融合结果越"平滑"。K 越小，top 排名结果的权重越高。

### 3.7 Rerank 重排（`retrieval/reranker.py`）

**为什么需要 Rerank？**

RRF 混合后得到 top 10，但排序依据是 RRF 分数——这是一个**间接的排序信号**（基于两路检索排名的融合）。Rerank 使用 Cross-Encoder 对 (query, document) 对直接计算相关性分数，排序更准确：

```
Embedding-based retrieval:  query → 向量 → 近似搜索（快速但粗略）
Cross-Encoder Rerank:       (query, doc) → 注意力计算 → 相关性分数（慢但精确）
```

**实现细节：**
- 调用 DashScope 的 `gte-rerank` 模型（通过 httpx POST 到非 OpenAI 兼容端点）
- 请求体结构：
  ```json
  {
    "model": "gte-rerank",
    "input": {"query": "...", "documents": ["...", "..."]},
    "parameters": {"top_n": 5}
  }
  ```
- 失败处理：`try-except` 兜底，API 异常时直接返回原始顺序
- 为什么用 httpx 而非 OpenAI SDK？DashScope Rerank API 不是 OpenAI 兼容格式

### 3.8 语义缓存（`cache/semantic_cache.py`）

**设计目标：** 相同或高度相似的问题不重复调用 LLM，降低延迟和 API 成本。

**核心机制：**

```
提问 → embed → ChromaDB（余弦距离搜索）→ 距离 < 0.03？→ 命中 → 直接返回缓存的回答
                                                      ↓ 超过阈值
                                                   未命中 → 正常走 LLM → 写入缓存
```

**阈值 `threshold=0.03` 的含义：** 余弦距离 ≤ 0.03 视为语义相同。这是一个很严格的阈值，意味着只有几乎一模一样的问题才会命中。可以根据实际数据调大。

**缓存淘汰策略：** 超过 `max_entries=500` 时，按 `created_at` 排序淘汰最旧的条目。

**为什么用独立的 ChromaDB collection（而非独立的缓存系统）？**
- 复用已有的 ChromaDB 基础设施，不需要额外引入 Redis/Memcached
- ChromaDB 本身支持持久化和向量搜索，天然适合语义缓存场景
- 代价是每次缓存查询多一次向量搜索（但 ChromaDB 是本地嵌入式，延迟 < 10ms）

**特殊设计：** 上传新文档时调用 `cache.clear()` 清空所有缓存。因为知识库变了，旧缓存可能包含已过时的回答。

### 3.9 查询改写（`generation/llm.py` 的 `rewrite` 方法）

**为什么要查询改写？**

用户提问往往是口语化、带缩写、带指代的，例如：
- `"OA有哪些功能"` → `"优化顾问(OA)有哪些功能"`
- `"CSS可用区未多AZ这个是啥意思"` → `"云服务CSS可用区未多AZ，这个风险检查项是什么意思？"`

改写后的查询在向量空间中更容易匹配到相关文档，因为改写过程：缩写展开、术语规范化、问句补全。

**实现方案：** 用 LLM 自身来做改写，而非规则 + 词典。

- 模板中包含：术语映射表 + 推理规则 + few-shot 示例
- `temperature=0.1` 保证输出稳定性
- 失败兜底：LLM 调用失败时返回原问题

### 3.10 LLM 生成（`generation/llm.py`）

**同步版本 `generate()`：** 普通 RAG 场景，返回完整文本。

**流式版本 `generate_stream()`：** 用于 SSE 推送，逐个 token `yield`。

**System Prompt 设计：**
```
你是一个基于文档内容回答问题的助手。
请根据以下提供的文档片段来回应用户的问题。
如果你在文档中找不到相关信息，请诚实地说明你不知道。
不要编造信息。
```

核心原则：**禁止幻觉**（不编造信息）、**诚实**（不知道就说不知道）。

**参数配置：**
- `temperature=0.3` — 低温度保证事实性回答
- `max_tokens=1024` — 足够回答大多数文档相关问题

**为什么用 GLM-4.7（智谱）而非 OpenAI？**
图中项目实际配置是 "deepseek-v4-flash"（截图显示），但 README 中示例为 glm-4-plus。说明该项目设计为 **LLM 无关**——通过 OpenAI 兼容 SDK 调用，切换模型只需改 `.env` 配置的三行参数（`api_key`、`base_url`、`model_name`）。

### 3.11 Web 爬虫（`ingestion/web_loader.py`）

**BFS 策略抓取网页，关键限制：**
- `max_depth=20` 控制递归深度
- 只抓取 `text/html` 类型
- 去重基于 visited URL set
- 图片引用 `![...](...)` 被正则移除

**为什么需要 web_loader？**
在很多 RAG 场景中，知识源是 Wiki/在线文档/技术博客。如果先手动下载再上传，流程繁琐。WebLoader 允许用户直接输入 URL 自动抓取。

### 3.12 查询日志（`query_log.py`）

**SQLite 本地存储的查询日志，记录字段：**

| 字段 | 说明 |
|------|------|
| `original_question` | 用户原始输入 |
| `rewritten` | 改写后的查询 |
| `cache_hit` | 是否缓存命中 |
| `latency_ms` | 端到端延迟 |
| `vector_n / bm25_n` | 两路检索的 top N 设置 |
| `vector_raw / bm25_raw` | 两路各自召回的数量 |
| `final_count` | RRF 后最终结果数 |
| `src_vector / src_bm25 / src_both` | 最终结果的来源分布 |

**距离度量的选择：** VectorStore 使用的是 ChromaDB 默认的 L2 欧氏距离（创建 collection 时未指定 `hnsw:space`），而 SemanticCache 显式指定了 `cosine` 距离。为什么两处不一致？从功能角度看，语义缓存的命中判定需要阈值比较，cosine 距离范围 [0, 2] 有明确的物理含义（1.0 = 正交，0.0 = 完全相同），便于设定通用阈值。向量检索则更关心排序（而非绝对距离值），L2 和 cosine 在归一化 embedding 上排序结果等价（因为 `||a-b||² = 2 - 2cos(a,b)` 当向量为单位长度时）。但如果 embedding 未归一化，两种度量会产生不同排序——这是一个值得注意的依赖点。

---

## 4. 选型决策全景

### 4.1 技术栈总览

| 层级 | 选型 | 备选方案 | 选择理由 |
|------|------|---------|---------|
| Web 框架 | **Flask** | FastAPI, Django | 同步架构够用、轻量、学习成本低 |
| 向量数据库 | **ChromaDB** | FAISS, Milvus, Pinecone | 全本地无需部署、pip install 即用、自动持久化 |
| Embedding | **Qwen text-embedding-v2** | text-embedding-v3, BGE, M3E | 开源免费、中文优化、OpenAI 兼容 |
| LLM | **GLM-4.7** | GPT-4, Claude, DeepSeek | 通过 OpenAI SDK 调用、可切换 |
| 稀疏检索 | **rank_bm25 + jieba** | Elasticsearch, Whoosh | 纯 Python 轻量、零运维 |
| Rerank | **DashScope gte-rerank** | Cohere Rerank, BGE-Reranker | 同厂商 Rerank 延迟更低 |
| Tokenizer | **tiktoken cl100k_base** | huggingface tokenizers | 社区标准、cl100k_base 兼顾中英文 |
| 分块 | **LangChain MarkdownHeaderTextSplitter** | 手写正则 / spaCy 分句 | 开箱即用、支持标题层级感知 |
| 配置 | **pydantic-settings** | python-dotenv, Dynaconf | 类型安全、自动从 .env 加载 |
| Web 爬虫 | **httpx + html2text** | requests + BeautifulSoup | httpx 支持连接复用、html2text 一键 Markdown |
| WSGI | **waitress** | gunicorn, uvicorn | Windows 兼容、支持 SSE 流式 |
| 前端 | **原生 HTML + JS** | React, Vue | 单页足够、零构建步骤 |

### 4.2 与 LangChain/LlamaIndex 的框架对比

**面试必问："为什么不直接用 LangChain？从头造轮子的意义是什么？"**

| 对比维度 | Tiny RAG | LangChain | LlamaIndex |
|---------|----------|-----------|------------|
| 透明性 | 每步显式可观测、可调试 | 大量封装默认行为，黑盒感强 | 模块化好于 LangChain |
| 依赖规模 | 12 个直接依赖 | 200+ 子包，版本冲突频繁 | 较精简但也有 50+ 依赖 |
| 定制灵活性 | 改 3-5 行代码即可 | 需自定义 class / callback | 可插拔，但需遵循框架约定 |
| 错误处理 | 基础级（需自行加强） | 框架级重试/fallback 内置 | 内置 |
| 学习曲线 | 几人天（全量代码 2000 行） | 几周（需理解 chain/agent/tool 等抽象） | 中 |

**回答要点：**
1. **透明度** — LangChain 的 `RetrievalQA` chain 封装了大量默认行为（prompt 模板、文档合并策略等），开发者容易忽略底层细节，出问题时难调试。Tiny RAG 每步都是显式代码，面试时可以准确说出每步的输入/输出格式
2. **依赖管理** — LangChain 生态有 200+ 子包且跨包版本约束复杂，tiny-rag 只用 12 个依赖，pip install 一次到位
3. **定制灵活性** — RRF 权重调优在 LangChain 中需要自定义 `EnsembleRetriever` 的权重参数，改写 prompt 需要自定义 `QueryTransformer`，而在 tiny-rag 中只需改 `data/config.yaml` 或 `config.py` 几行

**面试时的话术：**
> "选择不依赖框架是因为项目定位是学习型 RAG 系统，我需要透彻理解每一步的细节——从 chunk 如何分、embedding 怎么调、RRF 公式怎么写、到 SSE 怎么推。如果直接上 LangChain，这些细节都被抽象掉了。当然我也清楚自研的代价：错误处理、监控、大规模检索这些生产化能力需要自己补。如果要投产，我会考虑用 LlamaIndex 做底层编排（它模块化更好），但仍保留自定义的 rewrite 和 RRF 逻辑。"

### 4.3 选型分类详解

#### 为什么 ChromaDB 而不是 FAISS？

向量检索场景下，FAISS 的索引构建和搜索速度都优于 ChromaDB，但 FAISS 有以下短板：
1. **没有元数据存储** — 你需要额外的数据库记录每个向量属于哪个文档、哪个 chunk
2. **没有持久化方案** — 需要自己写 `faiss.write_index` / `faiss.read_index`
3. **没有客户端-服务端架构** — 在多进程场景下需要自己管理共享内存

ChromaDB 虽然底层使用了 HNSW 算法（与 FAISS 类似），但将向量存储、元数据管理、持久化、查询接口打包成了一个产品。对于中小规模（<100 万向量）场景，性能差异可以忽略。

#### 为什么 rank_bm25 + jieba 而不是 Elasticsearch？

ES 太重了——需要独立部署 Java 进程、配置索引映射、维护集群。rank_bm25 是纯 Python 实现，pip install 即用，对于几千个文档的 RAG 场景完全够用。

代价是需要自己维护分词和索引重建。当前实现是**全量重建**——每次增加文档，整个 BM25 索引重新构建。这在百级 chunk 场景只需几毫秒，但到了万级需要优化为增量更新。

#### 为什么 SSE 而不是 WebSocket？

SSE（Server-Sent Events）是单工协议——服务器→客户端，天然适合 LLM token 推送场景。WebSocket 是全双工，协议更复杂（需要握手、心跳、帧编码），但在这个场景中我们不需要从客户端流式发送数据。

SSE 的实现极为简单——Flask 的 `Response(stream_with_context(generator()), mimetype="text/event-stream")`。

**兼容性问题：** SSE 需要通过 `fetch` API + `ReadableStream` 在前端消费，因为 EventSource API 不支持 POST 请求（无法传 `force_refresh` 等参数）。

---

## 5. 关键设计决策与权衡

### 5.1 BM25 全量重建 vs 增量更新

**选择的策略：** 每次 `add_document` 全量重建 BM25 索引。

```python
def add_document(self, doc_id, filename, chunks):
    self._chunks.extend(chunks)
    self._metadatas.extend(...)
    tokenized_corpus = [self._tokenize(c) for c in self._chunks]
    self._bm25 = BM25Okapi(tokenized_corpus)  # 全量重建
```

**权衡：** 实现简单但扩展性差。对当前项目（预计 < 1000 chunks）完全 OK。如果要支持大规模文档，需要改为增量方式——维护分词后的 token 列表，用 `BM25Okapi` 的 `add_document` 或改用 Elasticsearch。

### 5.2 语义缓存阈值选择

阈值 `threshold=0.03`（余弦距离）是一个极端保守的设定：

- 余弦距离范围 [0, 2]，越小越相似
- 0.03 ≈ 余弦相似度 0.97，意味着只有几乎完全相同的问题才会命中
- **为什么这么保守？** 如果误命中，LLM 会给出与问题语义略有不符的回答，在技术问答场景中这种错误非常明显
- **可优化方向：** 根据实际日志调整到 0.05-0.1，可以显著提升缓存命中率

### 5.3 查询改写用 LLM vs 规则引擎

**用 LLM 改写的优点：**
- 可以处理复杂语义变换（"这个是啥意思" → "是什么"）
- 可以从上下文中推断未注册的缩写
- 维护成本低——改 Prompt 即可

**用 LLM 改写的缺点：**
- 增加一次 LLM 调用延迟（几百 ms）
- API 成本增加（但改写请求 token 很少，成本可忽略）
- LLM 可能过度改写或改变原意

**备选方案：** 先用规则做缩写展开（基于 TERM_MAP），如果规则匹配不到再 fallback 到 LLM。这种混合策略更经济。

### 5.4 Rerank 降级策略

Rerank 被设计为**可选组件**——没有 API key 时系统正常工作（只是少了一次重排）。这是一个重要的设计原则：**核心功能不依赖可有可无的增强组件**。

即使配置了 Rerank，API 调用失败时也会静默降级到原始 RRF 排序，而不是返回错误。这种"脆而不碎"的设计提升了系统健壮性。

### 5.5 已知技术债

面试中主动指出项目缺陷，比等面试官发现后追问更显深度：

1. **双重配置系统** — 同时存在 `pydantic-settings`（从 `.env` 加载 `Settings` 类）和 `_reload_config()`（从 `data/config.yaml` 加载检索参数）两套配置机制，`config.py` 同时维护模块级变量（`VECTOR_N` 等）和 `Settings` 类，改配置时容易漏同步
2. **无可删除文档 API** — `VectorStore` 有 `list_documents` 但无 `delete_document`，`BM25Retriever` 只有 `clear()` 全量清空。用户传错文件只能清空全部数据重来，此缺陷也导致上传时保守地 `cache.clear()` 全量清空缓存
3. **CLAUDE.md 配置漂移** — 文档中说缓存阈值 0.03，但 CLAUDE.md 写的是 0.07，与代码不一致
4. **无版本锁定** — `requirements.txt` 有些包只写了主版本号，不同时间 `pip install` 得到的依赖版本不同，可能引入兼容性问题
5. **缺少结构化日志** — 核心路径（`app.py` 的 upload/ask）完全没有日志输出，出问题只能靠 Python traceback 定位

### 5.6 模块级单例

```python
# 在 app.py 模块级别初始化
embedder = EmbeddingClient(...)
vector_store = VectorStore(...)
llm = LLMClient(...)
```

这种设计意味着这些实例在模块导入时初始化一次，后续所有请求复用同一连接。优点是轻量、无连接池管理代码；缺点是在多线程场景下需要确保客户端线程安全（OpenAI SDK 的客户端是线程安全的，ChromaDB 的 client 在多线程下需要小心）。

---

## 6. 评估管线与量化分析

### 6.1 评估管线概览

```
Pipeline: chunk_size → RRF 权重 → Embedding 模型 → Rerank 模型
```

每一环节的评估结果作为下一环节的输入配置。

### 6.2 chunk_size 选型

**评估指标：**
- **碎片率** — 指一个语义完整的段落被切到多个 chunk 的比例，越低越好
- **填充率** — chunk 内容占 max_tokens 的比例，越高越好

**结论：** `chunk_size=512`（碎片率 0%，填充率 87.9%）

更大的 chunk_size（如 1024）可能填充率更高（95%+），但会在检索时引入更多不相关内容，并且消耗更多 LLM 上下文。

### 6.3 RRF 权重评估

**评估方法：**
```
1. 准备测试文档 → chunk
2. 标注问答对（每个问题标注期望命中的 chunk_id）
3. 遍历权重组合：(alpha=7, beta=3), (5,5), (3,10), (1,10)
4. 对每组跑 Recall@K / MRR / 来源分布
```

**已知问题：** 当前测试集太小（13 chunks, 4 QA pairs），BM25 无独家命中，导致所有权重组合的 Recall/MRR 完全相同。需要补充 BM25 能独家命中的 QA pairs 才能区分不同权重的效果。

### 6.4 设计文档

项目有完善的 docs 目录：
- `docs/superpowers/specs/` — 各功能设计文档
- `docs/superpowers/plans/` — 实施计划

---

## 7. 面试问答集锦

### Q1: 你这个项目最核心的设计亮点是什么？

> **回答要点：**
> 双路检索 + RRF 混合融合是最核心的设计。单纯向量检索在处理专业术语、缩写、精确编号时召回率不足（因为 embedding 空间里"CCE"可能被映射到"存储"附近而非"云容器引擎"）。BM25 弥补了精确词匹配的需求，而 RRF 用一种无参数的方法将两路排序融合在一起。权重调优过程也体现了一个关键问题：如果两路检索结果数量差异大，RRF 的权重需要对数量做补偿——我们最终用了 alpha=3/beta=10 来平衡 12 条向量结果和 4 条 BM25 结果之间的贡献。

### Q2: 为什么选择 ChromaDB 而不是 FAISS 或 Milvus？

> **回答要点：**
> 项目定位是轻量级 RAG 系统。ChromaDB 的 PersistentClient 做到了零部署——pip install 后自动在本地创建持久化目录，不需要启动任何服务进程。对于中小规模场景（< 100 万向量），其底层 HNSW 索引的性能已经足够。FAISS 虽然索引更快，但缺少元数据管理和持久化方案，需要额外用 SQLite/JSON 记录每个向量对应的文档 ID 和 chunk 信息，增加了复杂度。Milvus 更是在项目初期的学习阶段完全不必要的运维负担。

### Q3: 语义缓存你们是怎么实现的？为什么用 ChromaDB 而不是 Redis？

> **回答要点：**
> 我们复用了已有的 ChromaDB 基础设施来做语义缓存。流程是：问题 embedding 在缓存 collection 中做余弦距离搜索，距离 < 0.03 时视为命中，直接返回之前 LLM 的回答。不用 Redis 是因为我们需要的是语义匹配而不是精确 key-value 匹配——Redis 的 KV 结构只能做 exact match，无法处理"CSS 可用区未多 AZ"和"CSS 可用区没配多 AZ"这种语义等价但字符串不同的情况。代价是多一次向量搜索，但本地 ChromaDB 的搜索延迟在个位数毫秒级别，完全可以接受。

### Q4: 你们的查询改写为什么用 LLM 而不是基于规则？

> **回答要点：**
> 我们最初确实用规则+词典做缩写展开，但发现两个问题：一是用户问法多样（"这个是啥意思"、"什么事"、"能否科普一下"），规则很难穷举；二是用户问题中隐含的实体关系需要语义理解才能解析，比如"CSS可用区未多AZ"是一个风险检查项的名称而不是三个独立关键词。用 LLM 做改写 + few-shot 示例可以同时解决这两个问题。当然代价是多一次 LLM 调用，但在我们的场景中改写 token 消耗极小（通常 < 50 tokens），而且我们用的 GLM 推理成本几乎可以忽略。作为优化方向，可以先做规则匹配，匹配不到再 fallback 到 LLM。

### Q5: RRF 权重 alpha=3, beta=10 这个值是怎么来的？为什么不是 1:1？

> **回答要点：**
> 这里有个关键洞察：RRF 的权重不能只看比例，还要考虑每路召回的条目数量。我们设 `vector_n=12, bm25_n=4`，如果 alpha=beta=1，向量路 12 条结果的总 RRF 分数远高于 BM25 路 4 条，BM25 结果实际上永远进不了 top 10。所以我们做了一个粗略的均衡计算：向量贡献 ≈ 12 × alpha，BM25 贡献 ≈ 4 × beta。为了让两路有接近的"总预算"，需要 beta > alpha。最终 alpha=3, beta=10 的比值大致补偿了 3:1 的数量差异。当然最科学的做法是用评估脚本在真实数据集上遍历权重组合做 grid search，但那需要足够大且标注好的测试集——这也是我们下一步要做的。

### Q6: 系统怎么处理 LLM 幻觉问题？

> **回答要点：**
> 我们从三个层面处理：
> 1. **System Prompt 明确约束** — "如果你在文档中找不到相关信息，请诚实地说明你不知道。不要编造信息"
> 2. **低 temperature=0.3** — 减少 LLM 的"创造性"，尽量让输出贴近检索到的文档内容
> 3. **上下文来源透明** — 前端会展示"找到以下相关内容"的片段列表，用户可以自行判断 LLM 的回答是否基于文档。SSE 的 context 事件在 token 生成之前推送，用户能看到 LLM 看到了什么文档

> 当然这些都不是完全消除幻觉的银弹。更激进的做法包括加上 citation 机制（要求 LLM 回答时标注引用了哪些句子）或者引入验证环节（用另一个 LLM 检查回答是否基于上下文）。

### Q7: 如果要支持 100 万篇文档，你觉得系统需要做什么改进？

> **回答要点：**
> 几个方向：
> 1. **BM25 全量重建 → 增量索引** — 当前每次 add_document 重建整个 BM25 索引，百万级文档不可行。换成 Elasticsearch 或建立分片索引
> 2. **ChromaDB → 分布式向量库** — ChromaDB 是单机嵌入式，百万级向量在搜索延迟和内存上都不够。可以迁移到 Milvus 或 Qdrant
> 3. **分块策略可能需要调整** — 当前 512 tokens 对于长篇文档会产生大量 chunk，需要评估是否调整 chunk_size 或改用更激进的去重策略
> 4. **增加异步处理** — 当前上传是同步阻塞，大文件会导致请求超时。需要改成先返回"处理中"状态，后台异步完成分块和向量化
> 5. **缓存策略升级** — 从 ChromaDB 语义缓存升级到 Redis + 语义缓存两层：Redis 做精确匹配（快速路径），ChromaDB 做语义匹配（慢速路径）

> **追问：当前架构最大的性能瓶颈是什么？**
> 同步阻塞架构。目前 waitress 默认 4 个工作线程，所有请求（embedding API 调用、ChromaDB 查询、BM25 重建）都是同步的，一个慢请求会阻塞整个线程池。更严重的是上传大文档时 `embedder.embed([all_chunks])` 一次性发全部 chunk 做 embedding，大文档（几千 chunk）会导致单次 API 请求超时。改进方向：异步 IO（FastAPI/quart）、embedding 分批并发（限制最大并行数避免 API 限流）、对相同 chunk 文本加 embedding LRU 缓存。

### Q8: 你们为什么用 SSE 而不是 WebSocket？

> **回答要点：**
> 流量模型决定了协议选择。LLM token 推送是单向的（服务器 → 客户端），不需要客户端往服务器发流式数据。SSE 在单方向场景下比 WebSocket 轻量得多：
> - SSE 基于标准 HTTP，不需要协议升级握手
> - SSE 自带断线重连机制（EventSource API）
> - SSE 的 Firewall 兼容性更好（WebSocket 可能被代理拦截）
> 
> 代价是我们需要用 `fetch` + `ReadableStream` 来消费 SSE（因为 EventSource 不支持 POST 请求），前端代码比用 EventSource 稍微复杂一些。

### Q9: Rerank 和 RRF 都是排序，为什么两个都要？

> **回答要点：**
> 两者解决的问题不同：
> - **RRF** 解决的是"如何融合两路异构检索的排名"——它是一个 **集成策略**，把向量召回的排名和 BM25 召回的排名合并成一个统一的排序。它的输入是两列排名，不需要计算原始特征。
> - **Rerank** 解决的是"如何用更精确的模型重新评估相关性"——它是一个 **精排阶段**，用 Cross-Encoder 对 (query, document) 对做深度语义交互，给出更准确的相关性评分。它的计算成本远高于向量检索。
> 
> 整个检索流水线是：**粗排（向量 + BM25）→ 融合（RRF）→ 精排（Rerank）**。前两步筛选出 top 10 候选，第三步用高精度模型重排 top 5。这种级联架构在成本和效果之间取得了平衡——只用一次 Rerank 处理 10 个候选，而不是对所有文档做 Rerank。

### Q10: 你对 tiny-rag 最不满意的地方是什么？如果重来会怎么改？

> **回答要点：**
> 最不满意的是**缺乏系统的测试数据**。RRF 权重评估脚本写好了但因为测试集太小（13 chunks, 4 QA pairs）无法区分不同权重的效果。整个评估管线（chunk_size → RRF → Embedding → Rerank）理论上应该用数据驱动的方式做选型，但实践中只有 chunk_size 选型有充分的数据支撑。

> 如果重来，我会在项目初期就建立一套**规模适中但高质量的标注测试集**（比如 100+ QA pairs，覆盖向量擅长和 BM25 擅长的问题各一半），这样每个环节的选型都可以定量评估，避免凭感觉配置参数。

> 另一个是 BM25 索引的全量重建问题——当前实现太 naive，虽然对 demo 场景够用，但要演进到生产级别必须改为增量更新。

### Q11: 你的系统在安全性方面有什么考虑？有什么隐患？

> **回答要点：**
> 当前系统在安全方面存在几个明显短板，如果投产需要优先解决：
> 1. **API key 管理** — `.env` 文件中明文存储，进程内存中也是明文，生产环境应托管到密钥管理服务（Vault、AWS Secrets Manager）
> 2. **文件上传无限制** — 没有大小限制也没有内容校验，恶意构造的超大文件或畸形 PDF 可直接撑爆内存或导致 PyMuPDF 崩溃
> 3. **SSRF 风险** — WebLoader 没有 URL 白名单和域名过滤，可能被利用扫描内网服务。改进方向：加白名单 + 解析 URL 后过滤私有 IP 段
> 4. **缺少请求鉴权** — 所有接口完全开放，没有 token 验证，可以在网络上被任意调用
>
> 但回归项目定位——这是一个学习型 RAG 系统，安全设计的深度和项目阶段匹配。面试时展示你知道这些隐患并能指出改进方向，比"系统没有安全问题"要好得多。

### Q12: 对于中文场景，你的系统做了哪些特殊处理？

> **回答要点：**
> 中文 RAG 有几个和英文不同的难点：
> 1. **分词** — 英文按空格分割即可，中文需要专门的分词器。我们用了 jieba 做 BM25 的分词，但是 jieba 对专业术语（如"CSS可用区"、"多AZ"、"etcd集群"）的分词准确率不如领域词典增强方案
> 2. **Token 效率** — tiktoken cl100k_base 对中文不高效，一个中文汉字 2-4 tokens，英文单词平均 1.3 tokens。意味着 `chunk_size=512` 实际容纳的中文信息量远少于英文。更优选择是用专门的中文 tokenizer 或者调大 chunk_size
> 3. **Embedding 的中英混合** — text-embedding-v2 中文效果好，但技术文档常见中英混写（"CCE集群的etcd版本升级"），混合内容 embedding 质量可能下降。目前没有专门处理这个问题
> 4. **查询改写的术语展开** — 中文技术文档缩写密集（OA/CCE/CSS），TERM_MAP 只有 3 条映射覆盖面明显不足。生产环境需要构建领域术语知识库或自动挖掘缩写-全称对

### Q13: 你这个系统怎么监控？怎么测试？怎么部署？

> **回答要点：**
> **监控：** 目前只有两处基础可观测性——`SemanticCache.get_stats()` 暴露缓存命中率/条目数，`QueryLog` SQLite 记录逐条查询延迟和来源分布。但缺乏时间序列指标（检索延迟 P50/P99、API 错误率、chunk 数变化趋势）、无 `/health` 健康检查端点、核心路径无结构化日志。如果要上线，至少需要加 Prometheus 格式的 `/metrics` 端点和 JSON 结构化日志。
>
> **测试：** 有 15 个测试文件覆盖了 tokenizer、chunker、loader、BM25、hybrid、reranker、vector_store、semantic_cache、app 路由。但存在盲区：(1) 核心 `/ask` SSE 路由全部 mock 了外部 API，无真正端到端集成测试；(2) 需要 API key 的测试在 CI 中被默认跳过；(3) 无 SSE 断连/超时等边界测试；(4) 无压力测试验证 chunker 和 BM25 重建的性能。
>
> **部署：** 完全无部署设施——无 Dockerfile、无 docker-compose、无 CI/CD。目前靠 `python -m src.tiny_rag.app` 裸启动。`waitress` 的 worker 数和超时硬编码在 `app.py` 中不可配。改进至少需要一个 Dockerfile 固化和 `.env.example` 模板。

---

## 附录：关键代码索引

| 功能 | 文件 | 行数 |
|------|------|------|
| Flask 路由 & 主流程 | `src/tiny_rag/app.py` | 290 行 |
| 配置管理 | `src/tiny_rag/config.py` | 103 行 |
| 文档加载 | `src/tiny_rag/ingestion/loader.py` | 43 行 |
| Markdown 分块 | `src/tiny_rag/ingestion/chunker.py` | 66 行 |
| Tokenizer | `src/tiny_rag/ingestion/tokenizer.py` | 21 行 |
| Embedding | `src/tiny_rag/embedding/client.py` | 27 行 |
| ChromaDB 向量库 | `src/tiny_rag/storage/vector_store.py` | 109 行 |
| BM25 检索 | `src/tiny_rag/retrieval/bm25.py` | 85 行 |
| RRF 混合融合 | `src/tiny_rag/retrieval/hybrid.py` | 46 行 |
| Rerank 重排 | `src/tiny_rag/retrieval/reranker.py` | 86 行 |
| 语义缓存 | `src/tiny_rag/cache/semantic_cache.py` | 126 行 |
| LLM 生成 | `src/tiny_rag/generation/llm.py` | 117 行 |
| Web 爬虫 | `src/tiny_rag/ingestion/web_loader.py` | 113 行 |
| 查询日志 | `src/tiny_rag/query_log.py` | 94 行 |
| 前端界面 | `src/tiny_rag/templates/index.html` | 288 行 |
| RRF 评估脚本 | `scripts/rrf_eval.py` | 329 行 |
