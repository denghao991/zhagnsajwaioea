# WebLoader + Markdown 语义分块器 设计文档

日期: 2026-06-01

## 概述

为 tiny-rag 增加在线 Wiki 页面加载能力和 Markdown 语义分块能力。解决两个问题：

1. 目前只能上传本地文件，无法从在线 Wiki 抓取内容
2. 目前按 token 数硬切，破坏语义完整性，不适合结构化 Markdown

## 1. 整体架构变化

```
当前:
  bytes → load_bytes/load_pdf → chunk_text(纯token滑动窗口) → embed → VectorStore + BM25

修改后:
  bytes → load_bytes/load_pdf → MarkdownChunker.chunk_text → embed → VectorStore + BM25
  URL  → WebLoader.load        → MarkdownChunker.chunk_text → embed → VectorStore + BM25
```

## 2. WebLoader

### 位置
新文件 `src/tiny_rag/ingestion/web_loader.py`

### 新依赖
- `html2text` — HTML 转 Markdown

### 类设计

```python
@dataclass
class PageResult:
    url: str
    markdown: str
    depth: int


class WebLoader:
    def __init__(self, max_depth: int = 20, request_timeout: float = 30.0):
        ...

    def load(self, start_url: str) -> list[PageResult]:
        """BFS 爬取，返回所有抓取到的页面列表"""
        ...
```

### BFS 爬取逻辑

1. **队列**: `deque[(url, depth)]`
2. **去重**: `set` 记录 visited URL（标准化后）
3. **深度**: depth <= max_depth（默认 20）
4. **每页流程**:
   - `httpx.get(url, timeout=30)`
   - 检查 Content-Type: 非 text/html 跳过
   - `html2text.HTML2Text().handle(html)` → markdown
   - `remove_image_markdown(markdown)` → 清除 `![...](...)`
   - 提取页面内所有 `<a href>` 链接
   - 如果 depth < max_depth，将链接加入队列

### URL 标准化

对所有 URL 做以下处理：
- 相对路径 → 绝对 URL（基于当前页 URL）
- 去掉 `#fragment`
- 只保留 `http://` 和 `https://`
- 去重（标准化后比较）

### 图片处理

`html2text` 转换后，用正则清除所有图片 markdown：
```
re.sub(r'!\[.*?\]\(.*?\)', '', text)
```

图片 markdown 被直接移除，不保留 alt 文本。

### 错误处理

| 场景 | 行为 |
|------|------|
| HTTP 超时/4xx/5xx | 日志警告，跳过该页，继续队列 |
| 非 text/html 响应 | 跳过 |
| URL 解析失败 | 跳过，日志警告 |
| 某页转换失败 | 跳过，继续队列 |
| 全部失败 | 返回空列表 |

### 新增 API 端点

`POST /upload_web`

Request:
```json
{
  "url": "https://wiki.example.com/page",
  "max_depth": 20
}
```

Response (200):
```json
{
  "pages": 5,
  "results": [
    {"url": "...", "chunks": 12},
    ...
  ]
}
```

处理流程：
1. WebLoader.load(url, max_depth) → list of PageResult
2. 对每个 PageResult，MarkdownChunker.chunk_text(page.markdown) → list of ChunkResult
3. 用嵌套循环分页 embed chunks
4. VectorStore.add_document per page + BM25Retriever.add_document per page
5. SemanticCache.clear()
6. 返回抓取页面数

## 3. MarkdownChunker

### 位置
改造已有 `src/tiny_rag/ingestion/chunker.py`

### ChunkResult 数据结构

```python
@dataclass
class ChunkResult:
    text: str
    heading_path: str = ""
    token_count: int = 0
```

### MarkdownChunker 类

```python
class MarkdownChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[ChunkResult]:
        """输入 Markdown，输出 list of ChunkResult"""
```

### 分块算法（4 步）

#### Step 1 — 标记保护区域

扫描全文，标记两类区域的**行范围**：

- **代码块**: 匹配 `^[ ]{0,3}\x60{3,}`（最多 3 空格缩进的 3+ 反引号）
  - 记录 fence 长度，用于匹配对应闭合 fence
  - info string（如 `python`）在行末
- **表格**: 连续以 `|` 开头的行（每行至少有 2 个 `|`）

代码块优先级高于表格：先标记代码块，再在剩余行中标记表格。

#### Step 2 — 按标题切分

在非保护区域内扫描标题：

- ATX 标题: `^(#{1,6})\s+(.+)$`
- Setext 标题: 不处理（YAGNI，Wiki 页面通常用 ATX）
- 标题层级追踪：

```python
stack = []  # [(level, title)]
...
遇到 # (level=1):  pop until stack[-1].level < 1, push
遇到 ## (level=2): pop until stack[-1].level < 2, push
...

heading_path = " > ".join(t.title for t in stack)
```

- 每个标题开始一个新的 section
- 前言内容（第一个标题之前）归入 `heading_path=""`，单独成 section

#### Step 3 — Section 大小管理

对每个 section 计算 token 数（tiktoken）：

| token 数 | 行为 |
|----------|------|
| < 100 | 与下一节内容合并为一个 chunk（仅一次，不递归）；如果下一节 = 100 合并后仍 < 100 也保留。末节孤子直接与前一节合并（仅一次） |
| 100 ~ chunk_size | 直接作为 1 个 chunk |
| > chunk_size | 在 section 内部按层级二次切分 |

**大 section 二次切分**：
```
首选: 按 \n\n (段落) 切，合并相邻小段
次选: 仍然太大 → 按 \n (行) 切
末选: 仍然太大 → token 滑动窗口（当前 chunker 逻辑，兜底）
```

不论哪级切分，都遵循：
- **不跨保护区域**（代码块/表格整块保留）
- **相邻子块之间应用 chunk_overlap**

Overlap 实现：
- 当大 section 二次切分产生 N 个子块时，相邻子块之间需要 overlap
- 对每个子块（除第一个），取前一个子块末尾 `overlap` token 的文本（encode → 取最后 overlap 个 token → decode），拼接到当前子块内容开头
- 示例: chunk_size=512, overlap=64。子块 A 文本 encode 后 300 tokens，取最后 64 tokens decode → "xxx"。子块 B = "xxx" + 子块 B 原文。B 最终编码约为 64+原 token 数

#### Step 4 — 输出

返回 `list[ChunkResult]`，每个 ChunkResult 包含 text、heading_path、token_count。

## 4. 现有文件改动

### chunker.py

- 保留 `chunk_text()` 函数签名兼容（内部调用 MarkdownChunker）
- 新增 `MarkdownChunker` 类
- `chunk_text()` 返回 `list[ChunkResult]`

### vector_store.py

- `add_document(doc_id, filename, chunks: list[ChunkResult], embeddings)`:
  - 从每个 `ChunkResult` 取 `.text` 作为 ChromaDB document
  - 向 metadata 写入 `heading_path` 字段

### app.py

- upload 流程：
  ```python
  chunks = chunker.chunk_text(content)  # 返回 list[ChunkResult]
  embeddings = embedder.embed([c.text for c in chunks])
  vector_store.add_document(doc_id, filename, chunks, embeddings)
  bm25_retriever.add_document(doc_id, filename, [c.text for c in chunks])
  ```
- 新增 `/upload_web` 路由
- 导入 WebLoader

### bm25.py / cache/semantic_cache.py

- **无改动** — BM25 只接收文本，SemanticCache 不感知 chunk 结构

## 5. 测试策略

### WebLoader

- Mock httpx 返回伪造 HTML，测试：
  - 基础 HTML→Markdown 转换
  - 链接提取（绝对/相对/片段/非http）
  - BFS 遍历顺序
  - max_depth 限制
  - URL 去重
  - 错误响应（404/超时/非HTML）跳过处理
  - 图片 markdown 被移除

### MarkdownChunker

- 真实 Markdown 测试用例：
  - 纯标题层级 → 正确的 heading_path
  - 代码块（含内部 # 不误判为标题）
  - 表格整块保留
  - 大 section 段落级切分
  - 相邻 chunk overlap 正确拼接
  - 前言内容（无标题前缀）
  - 小 section 合并
  - 末节孤子合并
  - 长列表无段落边界时行级/滑动窗口 fallback
  - 嵌入代码块的表格（代码块优先保护）

### API

- `/upload_web` 端点的 Flask test client 测试：
  - 缺少 url → 400
  - 成功调用 → 200 + 返回页数
  - Mock WebLoader 验证集成

## 6. 未纳入范围

- 图片多模态处理 — 直接丢弃 `![...](...)`
- PDF/Audio/Video 网页内容 — 跳过非 text/html
- 网页登录认证 — 仅处理公开可访问页面
- Rerank 对 heading_path 的利用 — 后续可优化
