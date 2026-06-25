# tiny-rag 系统架构与业务逻辑

## 一、系统定位

tiny-rag 是一个轻量级 RAG（Retrieval-Augmented Generation）系统，面向企业内部 OA/运维场景的知识问答。用户上传技术文档 → 系统建立索引 → 用户提问 → 双路检索 + LLM 生成回答。

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户界面 (Web)                         │
│              Flask + HTML/SSE 流式推送                   │
└────────────────────┬────────────────────┬───────────────┘
                     │                    │
                     ▼                    ▼
         ┌───────────────────┐  ┌──────────────────┐
         │  文档上传          │  │  提问             │
         │  POST /upload     │  │  POST /ask       │
         │  POST /upload_web │  │  SSE 流式返回     │
         └────────┬──────────┘  └────────┬─────────┘
                  │                       │
                  ▼                       ▼
         ┌───────────────────┐  ┌──────────────────┐
         │  预处理管线        │  │  检索生成管线     │
         │  加载 → 分块       │  │  改写 → 双路检索  │
         │  → Embedding       │  │  → RRF → Rerank  │
         │  → 存入双索引      │  │  → LLM → 流式输出 │
         └───────────────────┘  └──────────────────┘
                  │                       │
                  ▼                       ▼
         ┌───────────────────────────────────────────────┐
         │                 存储层                          │
         │  ChromaDB(向量) + BM25(关键词)                │
         │  + SQLite(查询日志) + SemanticCache(语义缓存) │
         │  + chunk_registry.json(chunk_id→原文离线映射) │
         └───────────────────────────────────────────────┘
```

---

## 三、文档上传流程

```
用户上传文件 (.md)
         │
         ▼
    ┌────────────┐
    │ 文件格式校验 │ ← 只允许 .md
    └─────┬──────┘
          │
          ▼
    ┌────────────────────────────┐
    │ MarkdownChunker             │ ← chunk_size=512*（可上线后调参）
    │ 分块                       │    overlap=64
    │ chunk_id = filename#N      │
    └─────┬──────────────────────┘
          │ 输出 ChunkResult 列表（text + token_count）
          │
          ├──────────────────────────────────────┬───────────────────┐
          ▼                                      ▼                   ▼
    ┌──────────────┐                    ┌──────────────────┐ ┌──────────────────┐
    │ Embedding     │                    │ BM25Retriever     │ │ chunk_registry   │
    │ 批量向量化     │                    │ jieba 分词 + BM25 │ │ （离线分析用）   │
    └──────┬───────┘                    │ 全量重建索引       │ │ dump_chunk_      │
           │                            └──────────────────┘ │ registry.py →    │
           ▼                                                  │ chunk_id: 全文   │
    ┌──────────────┐                                          │ JSON 映射表      │
    │ VectorStore   │                                          └──────────────────┘
    │ ChromaDB 入库  │
    │ metadata 保留  │
    │ doc_id,        │
    │ filename,      │
    │ chunk_index    │
    └──────┬───────┘
           │
           ├──────────────────┬──────────────────────┐
           ▼                  ▼                      ▼
    ┌──────────┐    ┌──────────────────┐    ┌──────────────────────┐
    │ Semantic  │    │ query_log 记录    │    │ 返回响应：           │
    │ Cache     │    │ 上传后清除旧缓存   │    │ {id, filename,      │
    │ .clear()  │    └──────────────────┘    │  chunks: N}          │
    └──────────┘                             └──────────────────────┘

    * = 带 * 的参数为「上线后调参」—— 需完整知识库上线后基于真实数据微调
```

### 关键设计点

- **分块策略**：MarkdownHeaderTextSplitter 按 `#`/`##` 标题先分，再用 RecursiveCharacterTextSplitter 按段落填充到 512 tokens
- **chunk_id 格式**：`{filename}#{chunk_index}`（如 `产品介绍#0`），贯穿向量存储、BM25 索引、查询日志和评估数据
- **双索引写入**：向量存 ChromaDB（PersistentClient），同时 jieba 分词后建 BM25Okapi 索引
- **WebLoader BFS**：从起始 URL 开始 BFS 爬取，httpx + html2text 转 Markdown，过滤图片链接，只抓 HTML 页面，非 HTML 跳过
- **BM25 全量重建**：每次 add_document 全量重建（数据量小时简单方案）
- **chunk_registry.json**：离线分析辅助文件，记录 chunk_id → chunk 全文映射，用于日志回放时查看具体命中内容
- **缓存清空策略**：上传新文档后全部清空，保证旧缓存不会返回过时内容

---

## 四、问答流程

```
用户提问（口语化，含缩写 "OA"、"HCSO"、"EVS" 等）
         │
         ▼
    ┌──────────────┐
    │ 查询改写      │ ← LLM（DeepSeek）展开缩写、补全术语
    │ LLM.rewrite()│     例："OA有哪些功能" → "优化顾问(OA)有哪些功能"
    └─────┬────────┘
          │
          ▼
    ┌──────────────┐
    │ Embedding     │ ← 改写后的问题向量化
    │ embedder     │
    └─────┬────────┘
          │
          ▼
    ┌──────────────────┐
    │ SemanticCache    │ ← 用问题向量检索缓存
    │ 缓存检查          │    余弦距离 < 阈值(0.03*) → 命中
    └─────┬────────────┘
          │
    命中──┤ force_refresh=true ────────────┬──────────────────────────────┐
          │ 直接跳过                    │                              │
          │ 未命中                      │                              ▼
          │                            │        ┌─────────────────────────┐
          ▼                            │        │ 命中后直接返回（跳过检索+LLM）│
    ┌──────────────────────────────────────┐   │                    │
    │          双路检索                     │   │                         │
    │                                      │   │ 1. query_log 记录:      │
    │  ┌─────────────────┐                 │   │    cache_hit=true       │
    │  │ 向量检索(Vector)  │  top 12*      │   │    vector_hits=[]       │
    │  │ ChromaDB         │  ← 余弦相似度  │   │    bm25_hits=[]         │
    │  │ 语义匹配          │               │   │    latency_ms           │
    │  └────────┬────────┘                 │   │                         │
    │           │                          │   │ 2. SSE 流式推送缓存回答  │
    │  ┌────────▼────────┐                 │   │    event: context(来源) │
    │  │ RRF 合并         │  k=60          │   │    event: token(逐字)   │
    │  │ alpha=7, beta=3  │  top 10        │   │    event: done          │
    │  └────────┬────────┘                 │   │      {cached: true}     │
    │           │                          │   └─────────────────────────┘
    │  ┌────────▼────────┐                 │
    │  │ BM25 检索        │  top 4*        │
    │  │ jieba分词+BM25  │  ← 关键词匹配  │
    │  │ 精确匹配         │                │
    │  └─────────────────┘                 │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────┐
    │ Rerank 重排       │ ← DashScope gte-rerank
    │ cross-encoder     │    对 top 10 逐对打分
    │ 语义精排          │    取 top 5
    └─────┬────────────┘
          │
          ▼
    ┌──────────────────┐
    │ LLM 生成          │ ← 改写后的问题 + 检索片段 → 生成回答
    │ generate_stream  │    流式 SSE 推送
    └─────┬────────────┘
          │
          ├────────────────────────────────────────┐
          ▼                                        ▼
    ┌──────────────────┐                  ┌──────────────────────────┐
    │ SSE 流式推送客户端 │                  │ 后处理（generate_and_cache）│
    │ event: context   │                  │                          │
    │ event: token     │                  │ 1. 完整回答存入 Semantic  │
    │ event: done      │                  │    Cache（key=改写后向量）│
    └──────────────────┘                  │                          │
                                          │ 2. 记录 query_log:       │
                                          │    • vector_hits:        │
                                          │      [filename#N, ...]   │
                                          │    • bm25_hits:          │
                                          │      [filename#M, ...]   │
                                          │    • 来源分布:           │
                                          │      src_vector/bm25/both│
                                          │    • latency_ms          │
                                          │    • cache_hit=false     │
                                          └──────────────────────────┘

    * = 带 * 的参数为「上线后调参」
```

### SSE 推送协议

三种事件类型：
```
event: context    ← 检索到的文档片段（供前端展示来源）
event: token      ← LLM 逐 token 流式输出
event: done       ← 结束事件，附带来源列表 + 元数据（是否缓存命中、改写前后对比等）
```

### 缓存策略（详细设计见第五章）

缓存的目标

  RAG 里做缓存的核心逻辑是：同样的问题 + 同样的文档 → 应该产生同样的回答。 这是一个高度确定性的场景（对比通用对话），所以缓存特别有效。

具体三层目标：
  1. 降低延迟（最直接） — LLM 生成要 2-5 秒，缓存命中 50ms 搞定。做 RAG 的用户等待是最核心的体验矛盾。
  2. 降成本（最实在） — 检索管线里：LLM 调用 + Embedding API + Rerank API 都是要花钱的。缓存命中一次，省掉整条管线。
  3. 一致性（最容易被忽略） — 两个同事问"OA有监控告警吗"，如果一个人得到的是缓存回答另一个人是 LLM
缓存的负面作用
  1. 语义误命中（最隐蔽）
     问"VPN怎么配置"和"VPC怎么配置"，embedding 距离可能很近。缓存命中后用户看到的是 VPN 的回答但以为是 VPC 的——看起来完全合理但其实是错的。
  用户发现不了，因为答案格式是对的。
     解决
     初始时,我们设置的余弦相似度很严格,0.03,避免功能刚上线就损害用户体验,并且提供了重新生成入口,不走缓存的机制,重新生成也不会覆盖缓存,语义匹配可能出现近似情况
     此外我们日志会记录缓存命中的情况以及匹配的向量和bm25命中情况,我们后续会将这些虽然通过了rag系统还是到oncall的这些问题的日志进一步分析
  2. 知识过时
     文档已经更新了，但缓存返回的还是旧文档的回答。这在 RAG 场景里比常规 Web 缓存更危险——因为 RAG 的核心卖点就是"基于最新的文档回答问题"。
        上传/删除/更新文档时,我们会清空缓存
        提供重新生成按钮给到用户跳过缓存
具体的设计
    key = 改写后的问题向量 ，原始问题完全不参与缓存匹配。
    value = LLM 回答 + 召回来源列表
    存入的是 cache.put(question=rewritten, answer=full_answer, sources=results)，其中：
        - answer：LLM 生成的完整回答文本
        - sources：RRF + Rerank 后的 top 5 结果（含 text, doc_id, filename, chunk_index 等）
        - question：改写后的文字版，存在 metadata 里备用
        - 不存 chunk 原文。 回答已经包含了需要的信息，chunk 只在检索阶段使用。
    未命中缓存则构造缓存,如果命中则直接返回带上chunkID
    如果点击重新生成,说明语义匹配了缓存,但是实际语义没有解决用户需求,直接走正常流程,新建一条缓存
    用户可以通过有用无用来删除缓存,其他仅通过更新文档的方式来删除缓存了
        缓存更新
          - 缓存不设最大上限,我们大概1月一个版本,就会更新一批文档,所有涉及更新文档的场景就会clear所有缓存
        
     
存在的问题 - 后续如何优化
    余弦距离 < 0.03 才认为两个问题的语义相同。这是一个偏保守的值（距离越小越严格），因为线上还没跑过，宁可不命中也不要误命中返回过时答案。属于"上线后调参"的参数，需要在真实流量下观察命中率和误中率来微调。
    force_refresh 就是用户不满意问到的缓存回答，要求重新用 LLM 生成一次，不做任何计数（不像之前的设计会累加 3 次后中毒）。

流程
question → llm.rewrite() → rewritten → embedder.embed([rewritten]) → question_vec
                                                                           ↓
                                                                cache.search(query_embedding=question_vec)
                                                                           ↓
                                                                cache.put(embedding=question_vec, ...)

注意：/ask 的 JSON body 支持 force_refresh 字段，传入 true 时跳过缓存直接调 LLM。
---

## 五、查询日志系统（QueryLog）

### 设计定位

SQLite 存储的只写日志系统，用于离线分析检索质量。只插入、不更新、不删除，自行连库查询。

### 数据表结构

```sql
CREATE TABLE query_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT NOT NULL,          -- UTC ISO 时间
    original_question  TEXT NOT NULL,          -- 用户原始提问
    rewritten          TEXT,                   -- 改写后的问题
    cache_hit          INTEGER DEFAULT 0,      -- 是否缓存命中

    latency_ms         INTEGER,               -- 请求总耗时（ms）

    vector_n           INTEGER DEFAULT 12,     -- 当时使用的检索参数
    bm25_n             INTEGER DEFAULT 4,      -- 方便后续回放时复现

    vector_hits        TEXT DEFAULT '[]',      -- JSON: 向量检索 top N 的 chunk_id[] 
    bm25_hits          TEXT DEFAULT '[]',      -- JSON: BM25 检索 top N 的 chunk_id[]
    final_count        INTEGER DEFAULT 0,      -- 最终送入 LLM 的片段数

    src_vector         INTEGER DEFAULT 0,      -- 最终结果中仅来自向量的数量
    src_bm25           INTEGER DEFAULT 0,      -- 最终结果中仅来自 BM25 的数量
    src_both           INTEGER DEFAULT 0,      -- 最终结果中同时命中的数量

    user_click         TEXT                    -- 预留：前端点击反馈 chunk_id
);
```
### 两条日志路径

| 场景 | cache_hit | vector_hits | bm25_hits | 触发点 |
|------|-----------|-------------|-----------|--------|
| 缓存命中 | true | [] | [] | cache.search() 返回结果后 |
| 全量检索 | false | 实际 chunk_id 数组 | 实际 chunk_id 数组 | LLM 生成完成后 |

### chunk_id 映射

chunk_id 格式为 `{filename}#{chunk_index}`（如 `产品介绍#0`），在分阶段生成。离线分析时通过 `scripts/dump_chunk_registry.py` 生成 `data/json/chunk_registry.json`，包含所有文档的 chunk_id → 全文映射，用于日志回放时还原命中内容。

### user_click（预留）

表中已预留 `user_click` 列，计划用于前端用户点击反馈。当前前端尚未实现该交互，但数据列已就绪，后续接入后可用于计算 CTR 等检索质量指标。

### 日志的两种用途

1. **在线调试**：查最近 50 条日志，看召回情况、响应耗时、缓存命中率
2. **离线评估**：导出一段时间的日志，与 chunk_registry.json 配合，分析检索盲区、调整 N 和 RRF 权重

---

## 六、核心模块详解

### 6.1 分块器（MarkdownChunker）

#### 分块流程

```
输入：原始文本
步骤：
  1. MarkdownHeaderTextSplitter 按 #/## 标题切分（保留标题层级信息）
  2. RecursiveCharacterTextSplitter 按段落填充（分隔符优先级：" " → "\n\n" → "\n"）
  3. 累积到 chunk_size，token 数用 tiktoken cl100k_base 计算
输出：ChunkResult(text, token_count)
```

分词器的作用: 是尽可能的将一篇长文档,保留语义的前提下合理切分,失去语义/语义过多的chunk会增加误召回率
描述: 我们业务的文档只有markdown格式,由于是内部文档,文档相对聚焦,我们直接复用了langchain的最佳实践，先按照标题切分,
标题切分后如果超过了chunkSize,则按照标点符号优先级递归切分
    MarkdownHeaderTextSplitter 负责按标题边界切分(只切分#,##,最大程度保留语义,避免片段太碎)
    RecursiveCharacterTextSplitter(按照段落,行来进行切分)
    先按最高优先级分隔符把文本切成多个片段。
    遍历每个片段：
        若片段长度 ≤ chunk_size：直接尝试放入当前 chunk（如果放不下会关 chunk 并新建）。
        若片段长度 > chunk_size：丢弃当前分隔结果，对该片段用下一个优先级的分隔符重新切，然后递归处理那些子片段。
        ```
        1. 用当前优先级最高的分隔符（比如 \n\n）把文本切开。
        2. 对每个片段，用 length_function（即 tiktoken）计算 token 数。
        3. 如果片段 token 数 ≤ chunk_size：加入当前 chunk，累积 token 数。
            当累积超过 chunk_size 时，当前 chunk 关闭，片段进入下一个 chunk。
        4. 如果片段 token 数 > chunk_size：降级到下一个优先级的分隔符，递归执行步骤 1-3。
        5. chunk_overlap 控制相邻 chunk 之间的重叠 token 数：关闭当前 chunk 时，保留最后 overlap 个 token 作为下一个 chunk 的前缀。
        ```
    切分完成后会将Document的metadata取出h1和h2,拼接到document content之前,补充语义
#### chunk_size 的作用
---
  1. 离线静态评估（已有，可扩展）

  当前 chunk_size_eval.py 做的 fill rate / fragmentation rate 是第一步筛选，筛掉明显不合理的尺寸：

  ┌────────────────────┬────────┬──────────────────────┐
  │        指标        │ 合格线 │         含义         │
  ├────────────────────┼────────┼──────────────────────┤
  │ Fill Rate P50      │ ≥ 70%  │ 过半块不浪费 token   │
  ├────────────────────┼────────┼──────────────────────┤
  │ Fragmentation Rate │ ≤ 5%   │ 极少文档被切得稀碎   │
  ├────────────────────┼────────┼──────────────────────┤
  │ Fill Rate P90      │ ≥ 85%  │ 极端情况也不严重浪费 │
  └────────────────────┴────────┴──────────────────────┘

  建议：把这个做成每次数据更新后自动跑的 CI 检查，输出一张表：
    1. 填充率, 找到中位数的chunk 占总chunk的百分比, 至少有一半的chunk的填充率高于中位数的填充率,
        代表chunk size设置的合理性,越大越好,但是边际递减
    2. 碎片率,碎片率,大概率不包含完整的语义信息,如果召回又会占召回名额,干扰rerank和最终的答案
    3. P75 P50 P90 -> 看实际的 50 75 90 分位实际上的chunk size是多少,他代表xx%的chunk size都小于Pxx的值
        例如P90 如果太小则代表碎片过多,如果p50
    4. 叠加人工召回

    但这个逻辑有一个陷阱——它只考虑静态指标，不考虑检索质量。你现在项目里的数据也印证了这一点：静态上 128 最优（fill_rate 79.7%），但检索评估（MRR）上 
  384 更好。所以这套静态评估的真实价值是"筛选"而非"决策"——它帮你排除明显不合理的 chunk_size（比如 1024 碎片率 0% 但 fill_rate 只有 20%
  这种），然后把候选列表交给检索评估去决选。


  chunk_size | fill_rate_p50 | frag_rate | pass?
  128        | 79.7%         | 0.0%      | ✅
  256        | 65.2%         | 2.1%      | ⚠️
  384        | 67.7%         | 2.1%      | ⚠️
  512        | 48.5%         | 2.1%      | ❌

  这块你已经跑完了，128 在静态指标上最优。

  ---
  2. 检索质量评估（关键一步）

  静态指标是 proxy，真正决定 chunk size 的是检索质量。做法：

  固定语料 → 按不同 chunk_size 分别建索引 → 用同一组问题查 → 对比 MRR/Recall@K

  你已有的 hybrid_compare.py 就是干这个的——从仓库里的结果看，chunk_size=384 在 RRF alpha=7/beta=3 下 MRR=0.7234 最高。这说明 384 虽然 fill rate 不如
   128，但检索质量更好。

  生产建议：把这套流程做成：
  1. 准备 50-100 条领域标注 question-answer 对（golden set）
  2. 每个候选 chunk size 跑一轮 eval → 输出 MRR/Recall@5/Recall@10
  3. 选 MRR 最高的
  4. 如果 MRR 接近（差距 < 0.01），选 fill rate 更好的那一个（节省 token + 降低延迟）

  ---
  3. 端到端质量评估（生产上线前的最终验证）

  检索质量不等于最终回答质量。最后一步：

  用选定的 chunk_size + 完整的 RAG pipeline → 对 golden set 生成回答
  → 用 LLM-as-judge 或人工打分 → 确认质量达标

  这一步你目前还没有自动化工具，生产团队通常用 GPT-4/Claude 对回答做 pairwise 比较（A/B test）。

  ---
  生产推荐策略

  ┌──────┬───────────────────────────────┬────────────────────┐
  │ 阶段 │             方法              │        产出        │
  ├──────┼───────────────────────────────┼────────────────────┤
  │ 初选 │ fill rate + fragmentation     │ 筛掉 512+ 的大块   │
  ├──────┼───────────────────────────────┼────────────────────┤
  │ 决选 │ MRR/Recall on labeled queries │ 选出检索最优的尺寸 │
  ├──────┼───────────────────────────────┼────────────────────┤
  │ 终验 │ LLM 回答质量评估              │ 确认最终质量       │
  ├──────┼───────────────────────────────┼────────────────────┤
  │ 上线 │ 定下来后固化 chunk_size 配置  │ chunk_size=384     │
  └──────┴───────────────────────────────┴────────────────────┘

  对你这个项目而言：静态数据已表明 128 最优，但检索 eval 显示 384 的 MRR 更高。如果最终回答质量也过关，我建议 chunk_size=384，128
  作为候选备选。理由是：384 在每个块里放了更多上下文，LLM 生成时信息更完整，而且你用 RRF 做了混合检索，块大一点对 BM25 也有利。

  ---
  额外建议：monitor after deploy

  上线后持续观察：
  - 平均响应延迟 —— chunk 越大，LLM 输入 token 越多，延迟会上升
  - 缓存命中率 —— 块越大，语义重复的概率越低，缓存命中率可能会下降
  - 这两个指标如果恶化，回退到 256 作为中间值



### 6.2 Embedding（向量化）

```
EmbeddingClient(base_url, api_key, model)
  .embed(texts: list[str]) → list[list[float]]
```

- 当前：Qwen text-embedding-v2（1536 维）
- 待切换：text-embedding-v3（512 维，更便宜，更长上下文，R@1=0.786 与 v2 持平）
- 评估工具：`scripts/embedding_eval.py` + 14 条人工标注 QA pairs
- 调用方式：DashScope OpenAI 兼容接口

### 6.3 向量存储（VectorStore）

封装 ChromaDB:

```
.add_document(doc_id, filename, chunks, embeddings)
  → 构建文档分组写入 ChromaDB collection
  → metadata 保留 doc_id, filename, chunk_index, heading_path
  → chunk_id 格式：{doc_id}_chunk_{index}

.search(query_embedding, n_results)
  → 余弦距离检索 → 返回排序结果（含 distance, heading_path）

.list_documents()
  → 按 doc_id 分组去重列出

.delete_document(doc_id)  — 尚未实现（已知局限）
```

### 6.4 BM25 检索

```
BM25Retriever
  .add_document(doc_id, filename, chunks)
    → 入参 chunks 是 list[str]（纯文本，不含 ChunkResult 对象）
    → jieba 分词 → 全量重建 BM25Okapi 索引
  .search(query, n_results)
    → jieba 分词 → BM25 打分 → top N
    → 返回 doc_id, filename, chunk_index, text, score
  .clear()
    → 清空全部索引（预留，当前代码中未调用）
```

### 6.5 RRF 混合检索（核心）

```
rrf_merge(vector_results, bm25_results, n_results, alpha, beta)

RRF 分数公式：
  score = weight / (K + rank)
  其中 K=60, rank 从 1 开始

  向量第 r 名得分 = alpha / (60 + r)
  BM25 第 r 名得分 = beta / (60 + r)

最终分数 = 向量分数 + BM25 分数（同时被两边命中时叠加）
```

当前权重：alpha=7.0, beta=3.0（向量主导，BM25 起辅助叠加作用）

> **坑点**：当前 VECTOR_N=12, BM25_N=4 下，BM25 rank 1 得分 = 3/61 ≈ 0.049，向量 rank 12 得分 = 7/72 ≈ 0.097，意味着 BM25 独立命中的结果排不进前 10。BM25 只有在同时被向量命中时才能通过叠加得分进入最终结果。这是上线后需要重点调优的参数组合。

### 6.6 Rerank 重排

```
RerankClient(base_url, api_key, model)
  .rerank(query, documents, top_n)
    → DashScope gte-rerank cross-encoder
    → 对 query 和每个 document 逐对计算相关性
    → 按相关性重新排序 → 取 top 5
```

当 rerank API key 未配置时，直接取 RRF 结果的前 5 条。

注意与 EmbeddingClient（使用 OpenAI 兼容接口）不同，RerankClient 通过 httpx 直接调用 DashScope 的专有 Rerank 端点（非 OpenAI 兼容格式）。API 调用失败时（网络错误、JSON 解析错误、HTTP 错误码）会静默降级，返回原始顺序。

### 6.7 语义缓存（SemanticCache）

```
SemanticCache(persist_dir, threshold, max_entries)
  独立 ChromaDB collection（不与文档向量混用）
  使用 hnsw:space=cosine 做近似最近邻检索

  .search(query_embedding)  → 余弦距离 < threshold → 返回缓存
  .put(question, answer, embedding, sources, entry_id) → 存入
                              超限时淘汰最旧条目（按 created_at）
  .clear()                  → 文档更新后清空
```

### 6.8 LLM 生成

```
LLMClient(base_url, api_key, model)
  .generate(question, context) → 完整回答
  .generate_stream(question, context) → token 迭代器（SSE 使用）
  .rewrite(question) → 改写后的问题
```

当前模型：DeepSeek-v4-flash（OpenAI 兼容接口）

生成参数：temperature=0.3, max_tokens=1024
改写参数：temperature=0.1, max_tokens=128（低温度保证改写稳定性）

改写 Prompt 结构：
  1. 系统指令 + 缩写映射表（TERM_MAP，从 config.yaml 热加载）
  2. 检查项推理规则（REWRITE_PATTERN）
  3. Few-shot 示例（REWRITE_EXAMPLES）
  4. 用户原始问题

### 6.9 Tokenizer（分词工具）

```
tokenizer.count_tokens(text) -> int       # token 计数（cl100k_base）
tokenizer.encode(text)      -> list[int]  # 文本 -> token ID
tokenizer.decode(tokens)    -> str        # token ID -> 文本
```

使用 tiktoken cl100k_base 编码。被 MarkdownChunker 依赖用于分块时的 token 计数，统一各模块的 token 口径。

### 6.10 WebLoader（网页加载器）

```
WebLoader(max_depth=20, request_timeout=30)

.load(start_url, max_depth?)
  -> BFS 爬取：httpx 请求 -> html2text 转 Markdown -> 过滤图片
  -> 逐层提取 <a href> 链接继续爬，直到 max_depth
  -> 只爬 HTML 页面，非 HTML（图片/PDF 等）跳过
  -> 返回 list[PageResult(url, markdown, depth)]
```

在 upload_web 流程中，每个 PageResult 独立生成 doc_id，走完整的分块 -> Embedding -> 双索引写入管线。

---

## 七、配置体系

### 外部服务配置（.env）

| 变量 | 说明 | 当前值 | 上线前确认 |
|------|------|--------|-----------|
| llm_api_key | 大模型 API Key | DeepSeek | ✅ |
| embedding_model | Embedding 模型 | text-embedding-v2 | 🔄 待切 v3(dim=512) |
| dashscope_api_key | 阿里云 DashScope Key | — | ✅ |
| rerank_llm_api_key | Rerank API Key | DashScope | 为空时跳过重排 |

### 检索参数（data/config.yaml）

系统启动时从 `data/config.yaml` 热加载（模块级 `_reload_config()` 在 import 时执行），覆盖 pydantic-settings 的默认值。修改 YAML 后重启服务生效。

| 参数 | 当前值 | 状态 | 说明 |
|------|--------|------|------|
| chunk_size | 512 | ✅ 已评估确认 | 碎片率 0%，填充率 87.9% |
| chunk_overlap | 64 | ⬜ 默认值 | 尚未系统性评估 |
| VECTOR_N | 12 | ⏳ **上线后调参** | 需完整知识库，扫饱和拐点 |
| BM25_N | 4 | ⏳ **上线后调参** | 需完整知识库，与 alpha/beta 联动 |
| alpha | 7.0 | ✅ 已评估确认 | 14 条 QA pairs 上 Recall/MRR 最优 |
| beta | 3.0 | ✅ 已评估确认 | 14 条 QA pairs 上 Recall/MRR 最优 |
| VECTOR_N + BM25_N | — | ⏳ **需完整库验证** | 当前 32 chunks 评估集太小，N 扫拐点无意义 |
| cache_threshold | 0.03 | ⏳ **上线后验证** | 需观察线上命中/误中率后微调 |
| cache_max_entries | 500 | ✅ 默认值 | 超出后淘汰最旧条目 |

YAML 结构示例（data/config.yaml）：
```yaml
cache:
  threshold: 0.03
  max_entries: 500
retrieval:
  vector_n: 12
  bm25_n: 4
  alpha: 7.0
  beta: 3.0
term_map:
  OA: 优化顾问(OA)
  CSS: 云服务CSS
  CCE: 云容器引擎CCE
```

---

## 八、评估与优化

### 已完成的优化

| 阶段 | 方法 | 结论 |
|------|------|------|
| chunk_size 选型 | 6 种候选，用 fill rate/fragmentation 评估 | 512 最优 |
| Embedding 选型 | v1/v2/v3(dim=512/1024) 对比 14 条 QA | v3(dim=512) 最优，R@1=0.786 |
| RRF 权重 | 14 条 QA pairs，4 组权重对比 Recall@K/MRR | alpha=7/beta=3 最优 |
| 日志系统 | vector_hits/bm25_hits(JSON 数组)，预留 user_click | 支持离线 replay |
| 查询改写 | LLM 展开缩写（OA → 优化顾问） | 3/14 条被改写，消除缩写歧义 |

### 评估指标

```
Recall@K：期望 chunk 是否出现在前 K 条结果中（K=1,3,5,10）
MRR（Mean Reciprocal Rank）：第一个期望 chunk 排名的倒数均值
来源分布：最终结果中来自向量/BM25/共同命中的比例
碎片率：token < 50 的 chunk 占比，用于 chunk_size 筛选
填充率：token 中位数 / chunk_size，表示空间利用率
```

### 待完成

| 项目 | 依赖 | 优先级 |
|------|------|--------|
| 切换 embedding 到 v3(dim=512) | 修改 .env + 验证 | 高 |
| 上传完整知识库 | 用户操作 | 高 |
| 基于完整库扫 VECTOR_N / BM25_N 饱和拐点 | 完整库 | 中 |
| 基于完整库复验 RRF 权重 + Rerank 评估 | 完整库 | 中 |
| cache_threshold 调优 | 线上流量 | 低 |
| user_click 前端接入 | 产品决策 | 低 |

---

## 九、已知局限性

1. **评估集小**：14 条 QA pairs 覆盖有限，调参可能不泛化
2. **BM25 被权重压制**：当前 VECTOR_N=12 / BM25_N=4 下 BM25 独立召回贡献为零（RRF 数学上无法进入 top 10）
3. **N 未确定**：VECTOR_N 和 BM25_N 的饱和拐点依赖完整知识库规模，当前 32-chunk 评估集无法给出结论
4. **BM25 全量重建**：文档量大时效率低（每次 add_document 重建全量索引）
5. **同步阻塞架构**：Flask + waitress 同步模型，高并发需额外扩容
6. **安全**：无认证鉴权、限流、防注入
7. **监控**：仅 SQLite 日志，无实时指标（Prometheus/Grafana）

---

## 十、技术栈一览

| 层次 | 技术 | 选型原因 |
|------|------|----------|
| Web 框架 | Flask | 轻量简单 |
| WSGI 生产 | waitress | 支持 SSE，跨平台 |
| 向量数据库 | ChromaDB (PersistentClient) | 本地化，无需部署服务 |
| Embedding API | DashScope text-embedding-v2/v3 | 中文优化，OpenAI 兼容 |
| LLM API | DeepSeek-v4-flash（OpenAI 兼容） | 低延迟，性价比高 |
| Rerank API | DashScope gte-rerank | cross-encoder 重排 |
| 关键词检索 | jieba + rank_bm25 | 中文分词成熟方案 |
| 分词工具 | tiktoken (cl100k_base) | 统一 token 计数 |
| 日志 | SQLite（query_log） | 零依赖，够用 |
| 离线分析 | chunk_registry.json（dump_chunk_registry.py） | 日志回放 + 检索质量分析 |
| 缓存 | ChromaDB 独立 collection | 复用已有组件 |

---

## 附录：上线后调参清单

以下参数无法在评估集上确定最终值，需要完整知识库上线后基于真实数据和用户反馈逐步调整：

| 参数 | 当前值 | 调参方法 | 判定标准 |
|------|--------|----------|----------|
| VECTOR_N | 12 | 在完整库上扫描 Recall@K 饱和拐点 | 超过 X 后 Recall@10 不再提升 |
| BM25_N | 4 | 与 VECTOR_N + alpha/beta 联动调整 | 来源分布中 src_bm25 > 0 |
| alpha/beta | 7/3 | 需在完整库上复验（当前 14 条 QA 可能不泛化） | Recall@K + MRR 最优 |
| cache_threshold | 0.03 | 观察线上命中率 vs 误中率 | 命中率 > 50% 且误中率 < 5% |
| chunk_overlap | 64 | A/B 测试：不同 overlap 下检索质量 | Recall@K |
