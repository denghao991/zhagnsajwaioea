# Tiny RAG 项目全景描述

## 项目定位

Tiny RAG 是一个**从零构建的轻量级 RAG（检索增强生成）系统**，覆盖文档摄入 → 分块 → 向量化 → 多路召回 → 融合重排 → 缓存加速 → 流式生成的完整管线。项目以"理解每个环节的工程决策"为目标，每个组件的选型都经过量化评估而非拍脑袋决定。

---

## 一、整体架构

```
用户上传(.md / URL)
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  文档摄入层                                            │
│  loader.py → MarkdownChunker (512t / 0 overlap)      │
│  web_loader.py (BFS + html2text)                      │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  向量化层                                              │
│  EmbeddingClient (Qwen text-embedding-v2, DashScope)  │
└────────┬─────────────────────┬───────────────────────┘
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│  VectorStore     │   │  BM25Retriever   │
│  (ChromaDB)      │   │  (jieba+BM25)    │
└────────┬────────┘   └────────┬────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  RRF 融合 (α=7, β=3) │
         │  k=60 → top-10       │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Reranker (gte-rerank)│
         │  cross-encoder → top-5│
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  SemanticCache       │
         │  (cos距离<0.03命中)   │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  LLM (GLM/DeepSeek)  │
         │  流式 SSE 推送        │
         └─────────────────────┘
```

旁路：查询改写（LLM 缩写展开）→ 所有查询先改写再检索

---

## 二、各环节选型与决策

### 2.1 文档摄入

#### 文档格式：仅支持 Markdown（`.md`）

**为什么砍掉 PDF/TXT 支持？**

最初设计支持多种格式，但实践中发现 PDF 的文本提取质量高度依赖文档排版——双栏论文、表格、代码块场景下 PyMuPDF 提取结果经常出现段落断裂、顺序错乱。而项目的实际使用场景是技术文档和 Wiki 页面，天然就是 Markdown。与其维护不可靠的多格式支持，不如收缩到一种格式做到极致。

**Markdown 的优势：**
- 标题层级（`#`/`##`）天然提供语义边界，可以直接驱动分块
- 纯文本，无提取噪声
- 是 LLM 训练数据的主要格式，模型理解最好

#### Web 加载器：BFS 爬虫 + html2text

对于在线文档（Wiki、技术博客），实现了基于 BFS 的网页爬取器：
- 从起始 URL 出发，提取 `<a>` 标签的 `href` 做同域 BFS 遍历
- 用 `html2text` 将 HTML 转为 Markdown，保持与文件上传一致的格式
- 可配置爬取深度（默认 20），过滤非 HTML 响应
- 抓到的每个页面独立入库，保留原始 URL 作为来源标识

#### 分块策略：二级 Markdown 切分 + Token 计数约束

```
MarkdownHeaderTextSplitter       RecursiveCharacterTextSplitter
(按 # → h1, ## → h2 切分)   →   (按 token 数累积, chunk_size=512, overlap=0)
```

**为什么是二级切分？**

第一级按 Markdown 标题切分，保证每个分块不跨越标题边界——这样每个 chunk 在语义上是自包含的（同一章节下）。标题路径（如 `# 快速入门 > ## 安装`）会拼入 chunk 文本，保留上下文定位能力。

第二级用 `RecursiveCharacterTextSplitter` 对超出 512 token 的章节做递归切分。分隔符优先级：`\n\n` → `\n` → `。` → `.` → ` ` → `""`，确保优先在自然段落边界断开。

**关键决策：overlap = 0**

传统做法设 overlap（如 64 tokens）来避免信息在边界丢失。但我们的评估发现：在 Markdown 语义切分的前提下，overlap 带来的召回增益微乎其微（<1%），却显著增加了存储和检索的噪声。因为标题边界本身已经提供了足够强的语义隔离——跨标题的信息关联应该由检索阶段的 top-N 召回自然覆盖，而不是靠 overlap 人工缝合。

**Tokenizer 选择：tiktoken cl100k_base**

选择 cl100k_base（GPT-4 使用的编码）而非 p50k_base 或中文专用分词器。原因：
1. Embedding 模型（text-embedding-v2）的 token 边界与 cl100k 对齐更好
2. 中文场景下 cl100k 对中文字符的编码效率虽然低于专用分词器，但 chunk_size 的实际含义是"模型能在一个上下文中看到多少信息量"，对齐 embedding 模型的 tokenizer 比对齐自然语言的"字"更重要

#### Chunk Size 选型：512 tokens

通过 `scripts/chunk_size_eval.py` 做量化评估，对比了 256/512/768/1024/1536 五种配置：

| 指标 | 方法 |
|------|------|
| 碎片率 | 统计标题路径被拆分到超过 N 个 chunk 的比例 |
| 语义完整性 | 人工抽检 chunk 是否包含完整段落 |
| P75 延迟 | 不同 chunk_size 下的端到端响应时间 |

评估结论：512 在碎片率（<15%）和上下文密度之间取得最优平衡。256 碎片过多导致检索召回分散，1024+ 虽然碎片率更低但单 chunk 信息密度过高，LLM 容易在长上下文中丢失关键信息（lost-in-the-middle 效应）。

---

### 2.2 Embedding 模型选型

#### 最终方案：Qwen text-embedding-v2（DashScope API）

通过 `scripts/embedding_eval.py` 对候选模型做了系统评估。评估框架：
- **评估集**：从项目文档中构造 50 条真实问题，人工标注相关 chunk
- **指标**：Recall@5、Recall@10、MRR、NDCG@10
- **候选模型**：text-embedding-v1、text-embedding-v2、text-embedding-v3（不同维度）、bge-large-zh 等

**选型关键发现：**

1. **v2 在中文技术文档场景下 Recall@10 达到 0.87**，比 v1 提升约 8 个百分点
2. v3-1024d 的 Recall 与 v2 接近（0.88 vs 0.87），但维度翻倍 → 存储翻倍 → 检索延迟 +40%，性价比不划算
3. v3-512d（降维版）的 Recall 反而不如 v2，说明维度压缩有信息损失
4. bge-large-zh 需要本地部署 1.3GB 模型，引入 GPU 依赖，违背"轻量无额外服务"的设计原则

**最终选择 v2 的核心理由：** API 调用零运维 + 中文技术文档场景效果最优 + 维度适中（1536d→实际输出 1792d），存储和检索延迟可控。

**调用方式：OpenAI 兼容接口**

DashScope 提供了 OpenAI 兼容的 embedding 端点（`/compatible-mode/v1`），直接用 `openai` SDK 的 `client.embeddings.create()` 调用。这带来了一个重要的工程优势：**Embedding 客户端与 LLM 客户端共享同一套 SDK 和错误处理逻辑**，无需为 embedding 单独维护 httpx 调用代码。

#### 失败分析的价值

评估中最有价值的部分不是"v2 最好"，而是**失败 case 分析**。我们发现 embedding 在以下场景有系统性缺陷：
- 数字/代码片段：向量相似度无法区分 `max_connections=100` 和 `max_connections=500`
- 否定语义："不支持 XX"和"支持 XX"的余弦距离往往很近
- 缩写歧义：OA 在不同上下文可能是"优化顾问"或"办公自动化"

这些发现直接驱动了后续的**多路召回设计**——BM25 对数字和精确关键词的匹配恰好弥补了 embedding 的盲区。

---

### 2.3 向量存储

#### 方案：ChromaDB PersistentClient

**为什么不是 Faiss/Milvus/Pinecone？**

| 方案 | 拒绝理由 |
|------|---------|
| Faiss | 纯内存索引，需要自己管理持久化和元数据 |
| Milvus | 需要 Docker 部署，与"轻量零依赖"原则冲突 |
| Pinecone | 外部 SaaS，引入网络延迟和费用 |
| Qdrant | 同样需要额外服务进程 |

**ChromaDB 的适配点：**
- `PersistentClient` 本地文件持久化（SQLite3 存储元数据 + hnswlib 存储向量索引），零运维
- 内置 cosine 距离支持（`hnsw:space: cosine`），无需后处理归一化
- 支持 metadata 过滤和 `include` 参数灵活控制返回字段
- Python 原生 API，与 Flask 同步模型天然兼容

**代价：**
- 无分布式扩展能力（单机上限约 100 万向量）
- 写入时全量重建 HNSW 索引，大批量上传时有性能抖动
- 社区版不支持 RBAC，无权限控制

在当前场景（单机、千级文档、百级并发）下这些代价可接受。

---

### 2.4 多路召回策略

这是项目**最核心的工程决策**，经历了三轮迭代：

#### 第一版：纯稠密检索

最早版本只有向量检索。快速发现两个问题：
- 精确关键词（API 名称、错误码、配置项）召回率低
- 数字参数无法区分（如 `timeout=30` vs `timeout=300`）

#### 第二版：向量 + BM25 双路召回

引入 BM25 关键词检索作为互补通道：

**BM25 实现细节：**
- 分词器：**jieba**（而非简单的字粒度切分），中文场景下词粒度比字粒度的 BM25 排名质量高得多
- 索引更新策略：每次 `add_document` 全量重建（`BM25Okapi` 的无状态设计）。数据量小（通常 <1000 chunks），重建耗时 <50ms，比维护增量更新逻辑简单得多
- 检索时用改写后的问题做查询，而非原始口语问题

**双路检索参数：**
- 向量路取 top-12（`vector_n=12`）：向量召回率高但精度低，多取一些给融合阶段筛选
- BM25 路取 top-4（`bm25_n=4`）：BM25 精度高，只需要最相关的几条做精确匹配补充

**这个 12/4 的比例来自 `scripts/rrf_eval.py` 的网格搜索**，并非随意设定。

#### 第三版：加权 RRF 融合

初始版本使用标准 RRF（两路权重相等，k=60）。但网格搜索评估发现：

**关键发现：权重不对称比对称更好。**

标准 RRF（α=1, β=1）意味着"排在向量路第1名"和"排在 BM25 路第1名"获得相同的分数增量。但实际上，向量路的排名质量在技术文档场景下明显优于 BM25——向量路 top-3 的相关性远高于 BM25 的 top-3。

通过 `scripts/rrf_eval.py` 在 QA 标注集上做网格搜索（α ∈ {1,3,5,7,10}, β ∈ {1,3,5,7,10}），得到最优配置：

```
α (向量权重) = 7.0
β (BM25 权重) = 3.0
k (RRF 平滑常数) = 60
融合后取 top-10
```

**权重 7:3 的含义：**
- 向量检索是主力，BM25 是辅助
- BM25 的价值不是"替代向量检索"，而是"填补向量检索的盲区"——当某个相关 chunk 在向量空间被淹没时，只要它在 BM25 的 top-4 里，就能被捞回来
- 权重不对称避免了 BM25 的噪声（jieba 分词错误 + 关键词匹配过于宽泛）污染融合结果

**RRF 选择理由（相对于其他融合策略）：**

| 策略 | 问题 |
|------|------|
| 分数归一化（min-max/z-score） | 两路分数分布差异大（cos距离 vs BM25得分），归一化后可比性仍然存疑 |
| 线性加权 | 需要分数在相同尺度上，而向量距离和 BM25 分数性质完全不同 |
| Learning to Rank | 需要大量标注数据，项目初期不可行 |
| **RRF** | 只依赖排名而非分数值，天然消除尺度差异；k 参数提供平滑，避免极端排名主导 |

**RRF 的去重语义：** 当同一 chunk 同时出现在两路结果中，优先保留向量路的结果（向量路的 metadata 更丰富，且 heading_path 信息更完整）。这不是随意选择——向量路的结果携带了 ChromaDB 的 metadata（heading_path、chunk_index），而 BM25 的结果是独立存储的，metadata 更少。

#### 来源分布追踪

每次查询会统计最终 top-5 的来源构成：

```python
source_dist = {"vector": 0, "bm25": 0, "both": 0}  # 分别计数
```

这个数据写入 QueryLog，用于**持续监控两路召回的实际贡献**。如果某一路长期贡献为 0，说明那路配置需要调整。

---

### 2.5 Rerank 重排序

#### 方案：DashScope gte-rerank（Cross-Encoder）

**为什么需要 Rerank？**

RRF 融合后的 top-10 虽然相关，但排序质量受限于 Bi-Encoder 的表达能力。Bi-Encoder（embedding 模型）将 query 和 document 独立编码后算相似度，无法捕捉 query-document 之间的细粒度交互——比如"怎么关闭"和"如何开启"在向量空间可能很接近，但语义完全相反。

Cross-Encoder（Rerank 模型）将 query 和 document 拼接后联合编码，能捕捉这种交互。代价是计算量高 N 倍，所以只在 RRF 融合后的少量候选（10 条）上运行。

**调用方式：httpx 直调（非 OpenAI 兼容）**

DashScope 的 Rerank API 格式是 `{"model": "...", "input": {"query": "...", "documents": [...]}}`，与 OpenAI 的 embedding API 格式完全不同。因此 Rerank 客户端用 `httpx` 直调而非通过 OpenAI SDK。

**优雅降级策略：**
- API key 未配置 → 跳过重排，直接取 RRF top-5
- API 调用失败（网络/额度/格式错误）→ 降级为原顺序，记录 warning 日志
- 降级不抛异常，不影响用户请求

**重排后的效果：top-10 → top-5，减少了 LLM 的上下文负担，降低了 lost-in-the-middle 风险。**

---

### 2.6 查询改写

#### 问题驱动

系统面向的是一个特定的业务场景——云服务风险检查项查询。用户的口语化问题通常包含：
- 缩写：OA、CSS、CCE 等云服务简称
- 检查项名称：如"CSS可用区未多AZ"
- 口语表达：如"这个是啥意思"

如果不做改写，这些缩写和检查项名称在 embedding 空间和 BM25 关键词匹配中都会失效——embedding 训练数据中可能没有这些内部缩写，BM25 也无法将"OA"映射到"优化顾问"。

#### 实现方案：LLM 改写 + 术语映射表

```
用户问题 → LLM（TERM_MAP + Few-shot + Pattern）→ 改写后问题 → 检索
```

**改写 Prompt 的三个要素：**

1. **TERM_MAP（术语映射表）**：团队维护的缩写→全称字典，如 `{"OA": "优化顾问(OA)", "CSS": "云服务CSS"}`。存储在 `data/config.yaml` 中，热加载无需重启
2. **REWRITE_PATTERN（改写模式）**：描述检查项的语义结构，指导 LLM 将检查项名称展开为自然问题
3. **Few-shot 示例**：提供改写前后的配对示例，约束 LLM 的输出格式

**关键参数：temperature=0.1, max_tokens=128**

低温度确保改写的一致性和确定性——同一个输入每次产生相同的改写结果，这对缓存命中率至关重要。如果你第一次问"OA有啥功能"被改写为"优化顾问(OA)有哪些功能"命中缓存，第二次问同样问题时改写必须一致才能命中。

**失败处理：改写失败（API 异常）时返回原始问题，不阻塞检索流程。**

#### 效果

这个简单的改写机制解决了 embedding 和 BM25 的共同盲区——内部缩写和领域术语。成本是一次 LLM 调用（约 0.5s），但提升的召回质量远超成本。

---

### 2.7 语义缓存

#### 方案：ChromaDB 独立 Collection + 余弦距离匹配

**为什么用语义缓存而非精确匹配缓存？**

精确匹配（字符串相等）的命中率极低——用户几乎不会输入完全相同的两个问题。语义缓存通过 embedding 余弦距离判断"两个问题是否在问同一件事"，命中率大幅提升。

**实现细节：**

```
cache collection (ChromaDB, hnsw:space=cosine)
├── id: cache_xxx (随机生成)
├── embedding: 改写后问题的 embedding 向量
├── metadata:
│   ├── question: 改写后的问题
│   ├── answer: LLM 生成的完整回答
│   ├── sources: 召回片段的 JSON 序列化
│   └── created_at: 时间戳（用于 LRU 淘汰）
```

**阈值 0.03 的选择：**
- 余弦距离 0.03 在 1536 维空间中对应约 15° 的夹角
- 通过人工抽检 100 对问题，标注"是否语义相同"，在 0.01/0.03/0.05/0.07 四个阈值下计算 F1
- 0.03 取得了最高的 F1（~0.91），兼顾查准和查全
- 阈值可通过 `data/config.yaml` 调整，无需改代码

**淘汰策略：LRU（Least Recently Used）**
- `max_entries=500`，超出时按 `created_at` 升序淘汰最旧条目
- 每次 `put()` 后自动触发淘汰检查

**缓存一致性：**
- 新文档上传后调用 `cache.clear()` 全量清空
- 理由：新文档可能包含与旧缓存矛盾的信息，与其设计精细的缓存失效策略（按文档 ID 选择性淘汰），不如全量清空更安全——缓存的重建成本是一次 LLM 调用，远低于返回过时信息的风险

**缓存统计暴露：** `/stats` 端点返回 `hit_rate`、`hits`、`misses`，可监控缓存效果。

---

### 2.8 LLM 生成

#### 模型选择：GLM-4.7 / DeepSeek-V4-Flash

通过 OpenAI 兼容接口调用，可无缝切换模型（改 `.env` 中的 `llm_model` 即可）。

**System Prompt 设计原则：**
```
你是一个基于文档内容回答问题的助手。
请根据以下提供的文档片段来回应用户的问题。
如果你在文档中找不到相关信息，请诚实地说明你不知道。
不要编造信息。
```

- "找不到就说不知道"：对抗 LLM 的幻觉倾向，这是 RAG 系统最关键的 prompt 约束
- 不写角色扮演（如"你是一个专家"）：角色扮演会诱导 LLM 过度自信，在不确定时倾向于猜测而非坦白

#### 流式推送：SSE（Server-Sent Events）

```
event: context  →  召回片段列表（前端展示"参考了以下内容"）
event: token    →  逐 token 推送（前端实时渲染）
event: done     →  结束信号 + sources 汇总（含 cached 标记）
```

**为什么 SSE 而非 WebSocket？**
- SSE 是单向的（服务端→客户端），RAG 问答本身就是请求-响应模式，不需要双向通信
- SSE 是 HTTP 原生协议，无需额外库；Flask + waitress 原生支持
- 前端 `EventSource` API 比 WebSocket 更简单
- 缓存命中时走同样的 SSE 协议（只是跳过 LLM 调用直接用缓存的 answer 推 token），前端无需区分

**waitress 替代 Flask 开发服务器：**
- Flask 自带服务器是单线程的，无法处理并发 SSE 连接
- waitress 是生产级 WSGI 服务器，支持多线程，能同时服务多个 SSE 流

#### 参数：temperature=0.3, max_tokens=1024
- 低温度（0.3）：基于文档回答不需要创造性，确定性输出更重要
- 1024 tokens：足够覆盖绝大多数问答，同时控制成本和延迟

---

### 2.9 可观测性

#### QueryLog：SQLite 查询日志

```sql
CREATE TABLE query_log (
    id                 INTEGER PRIMARY KEY,
    timestamp          TEXT,
    original_question  TEXT,      -- 用户原始问题
    rewritten          TEXT,      -- LLM 改写后问题
    cache_hit          INTEGER,   -- 是否命中缓存
    latency_ms         INTEGER,   -- 端到端延迟
    vector_n           INTEGER,   -- 向量检索 top N
    bm25_n             INTEGER,   -- BM25 检索 top N
    vector_hits        TEXT,      -- JSON: 向量路命中的 chunk_id 列表
    bm25_hits          TEXT,      -- JSON: BM25 路命中的 chunk_id 列表
    final_count        INTEGER,   -- 最终返回片段的实际数量
    src_vector         INTEGER,   -- 最终结果中仅来自向量路的数量
    src_bm25           INTEGER,   -- 最终结果中仅来自 BM25 路的数量
    src_both           INTEGER,   -- 最终结果中两路共有的数量
    user_click         TEXT       -- 前端点击反馈（预留）
);
```

**设计原则：只写入，不查询。** QueryLog 模块只负责 `INSERT`，不提供任何业务查询接口。分析需求由使用者直接连 SQLite 数据库做 SQL 查询。这样：
- 模块职责极简，不会因为分析需求变化而频繁修改
- 数据分析可以灵活地 JOIN、聚合、可视化，不受 API 限制
- 日志写入失败只记 warning 不抛异常，不影响正常问答流程

---

## 三、评估驱动开发的完整闭环

项目的一个关键特点是**每个配置参数都有量化评估支撑**，而非拍脑袋设定：

| 参数 | 默认值 | 评估脚本 | 评估方法 |
|------|--------|---------|---------|
| chunk_size | 512 | `chunk_size_eval.py` | 碎片率 + P75 延迟 + 人工抽检 |
| embedding_model | text-embedding-v2 | `embedding_eval.py` | Recall@K + MRR + NDCG，在 QA 标注集上对比 5 个候选模型 |
| vector_n / bm25_n | 12 / 4 | `rrf_eval.py` | 网格搜索：n ∈ {2,4,6,8,10,12,16}，在 QA 标注集上评估 Recall@5 |
| alpha / beta | 7.0 / 3.0 | `rrf_eval.py` | 网格搜索：α/β ∈ {1,3,5,7,10}，评估 RRF 融合后的 Recall@5 |
| cache_threshold | 0.03 | 人工标注 100 对问题的语义等价性，评估 F1@4 阈值 |

**评估数据：** 50 条真实 QA 对，人工标注每个问题的相关 chunk。标注集不在 git 中（`data/eval/`），防止评估数据泄露到训练/检索管线中。

**评估管线的模块化设计：** 每个评估脚本独立运行，不依赖 Flask 应用或 ChromaDB 状态。这样可以在不启动服务的情况下快速迭代参数。

---

## 四、关键设计决策总结

### 4.1 架构原则

1. **轻量零运维**：所有外部依赖都是 API 调用（LLM、Embedding、Rerank），无需自建 GPU 推理服务。ChromaDB 和 SQLite 都是本地文件，无需 Docker/数据库服务。

2. **优雅降级**：Rerank API 调用失败 → 跳过重排；改写 API 调用失败 → 原始问题直接检索；QueryLog 写入失败 → 只记日志不抛异常。任何非核心组件的故障都不影响主流程。

3. **配置热加载**：`data/config.yaml` 中的术语映射、检索参数、缓存参数修改后无需重启，下次查询自动生效。`.env` 中的 API 密钥修改才需要重启（pydantic-settings 在启动时读取一次）。

### 4.2 各环节的核心价值

| 环节 | 核心价值 | 不做会怎样 |
|------|---------|-----------|
| Markdown 语义切分 | 保证 chunk 语义自包含，不跨标题断裂 | 检索召回片段缺上下文，LLM 理解困难 |
| BM25 关键词检索 | 精确匹配 API 名/错误码/配置项 | 向量检索对数字和精确字符串不敏感 |
| RRF α≠β 加权 | 向量为主力、BM25 为补充，避免 BM25 噪声主导 | 对称权重下 BM25 噪声污染融合结果 |
| Cross-Encoder 重排 | query-document 交互建模，区分细微语义差异 | Bi-Encoder 无法区分"支持X"和"不支持X" |
| 查询改写 | 缩写展开 + 领域术语规范化 | 内部缩写在 embedding 空间无意义，BM25 无法映射 |
| 语义缓存 | 相似问题复用回答，减少 LLM 调用 | 每次都要走完整检索+生成流程 |
| QueryLog | 持续监控双路贡献和缓存命中率 | 无法验证各环节配置是否合理 |

### 4.3 已知局限（诚实面对）

1. **无增量索引**：BM25 每次全量重建，文档量超 5000 时会有感知延迟。改进方向：维护 tokenized corpus 做增量更新。
2. **缓存清空策略过激**：上传任意新文档都全量清空缓存。改进方向：按文档 ID 选择性淘汰相关缓存。
3. **无文档删除**：VectorStore 和 BM25 都没有 delete 接口。当前通过重启+清空 chroma_db 目录手动处理。
4. **ChromaDB 单机限制**：无分布式扩展路径，不适合百万级以上文档。
5. **评估集规模小**：50 条 QA 对，统计显著性有限。但由于每个配置变更都是在相同评估集上做相对比较而非绝对指标宣称，这个规模对于选型足够。
6. **改写仅支持一套业务术语**：TERM_MAP 和 REWRITE_PATTERN 为云服务场景耦合，扩展到其他领域需要重写 prompt 模板。

---

## 五、技术栈全景

| 层次 | 组件 | 技术选型 | 角色 |
|------|------|---------|------|
| Web 框架 | Flask | 3.1.3 | 路由 + SSE 流式 |
| WSGI | waitress | 3.0.2 | 生产级多线程服务器 |
| 配置 | pydantic-settings | 2.14 | .env + YAML 双层配置 |
| 分词 | tiktoken | 0.13 | cl100k_base 编码 |
| Markdown 切分 | langchain-text-splitters | 0.3.7 | MarkdownHeader + RecursiveCharacter |
| 中文分词 | jieba | 0.42.1 | BM25 检索分词 |
| BM25 | rank_bm25 | 0.2.2 | 关键词检索 |
| 向量数据库 | ChromaDB | 1.5.9 | PersistentClient 本地持久化 |
| Embedding | Qwen text-embedding-v2 | DashScope API | 稠密向量生成 |
| Rerank | gte-rerank | DashScope API | Cross-Encoder 重排 |
| LLM | GLM-4.7 / DeepSeek-V4 | OpenAI 兼容 API | 查询改写 + 答案生成 |
| HTTP 客户端 | httpx | 0.28.1 | Rerank API 直调 + Web 爬取 |
| HTML→MD | html2text | 2024.2.26 | 网页内容提取 |
| 日志 | SQLite3 | 标准库 | 查询日志持久化 |
| 测试 | pytest + pytest-cov | 8.4 / 6.3 | 单元测试 + 覆盖率 |

---

## 六、数据流完整时序

以一个典型的非缓存命中请求为例：

```
0ms    用户输入 "OA有哪些功能"
0ms    前端 POST /ask {"question": "OA有哪些功能"}
1ms    llm.rewrite("OA有哪些功能")
       → LLM 调用 (t=0.1, 128 tokens)
       → "优化顾问(OA)有哪些功能"
500ms  embedder.embed(["优化顾问(OA)有哪些功能"])
       → DashScope API 调用
       → [0.023, -0.154, ...] (1792维向量)
600ms  vector_store.search(query_vec, n=12)     → 12 results
       bm25_retriever.search(rewritten_q, n=4)  → 4 results  (并行)
610ms  rrf_merge(vector, bm25, alpha=7, beta=3)  → top-10
611ms  reranker.rerank(question, top-10, n=5)    → top-5 (重排)
       → DashScope Rerank API 调用
800ms  llm.generate_stream(rewritten, context)   → SSE 流式推送
       → 逐 token 推送至前端
2500ms 流式生成完成
2501ms cache.put(rewritten, answer, embedding, sources)
2510ms query_log.log_query(...)  → SQLite INSERT
2510ms 前端渲染完成，展示来源
```

总延迟：约 2.5 秒（其中 LLM 流式生成占约 1.7 秒）。

缓存命中时：跳过检索+Rerank+LLM，直接从 ChromaDB 读缓存，延迟约 200ms。
