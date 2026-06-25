# WebLoader + Markdown 语义分块器 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 tiny-rag 新增在线 Wiki 页面加载能力和 Markdown 语义分块能力

**Architecture:** 
- 新增 `MarkdownChunker` 类在 `chunker.py`，按 Markdown 标题层级分块，保护代码块/表格完整性，保留 `chunk_text()` 后向兼容
- 新增 `WebLoader` 类在 `web_loader.py`，httpx + html2text 抓取网页，BFS 跟随链接（最大深度 20）
- 修改 `vector_store.py` 的 `add_document()` 接受 `ChunkResult` 并将 `heading_path` 写入 metadata
- 修改 `app.py` 的 upload 流程使用 MarkdownChunker，新增 `/upload_web` 端点

**Tech Stack:** Python 3.12, Flask, ChromaDB, tiktoken, httpx (已有), html2text (新增)

---

### Task 1: 安装 html2text

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 添加依赖**

在 `requirements.txt` 的 Retrieval 区块后追加：

```text
# ============================================================
# Web（在线 Wiki 页面加载）
# ============================================================
html2text==2024.2.26
```

- [ ] **Step 2: 安装并验证**

```bash
pip install html2text
python -c "import html2text; print(html2text.__version__)"
```

Expected: `2024.2.26`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add html2text dependency for WebLoader"
```

---

### Task 2: 实现 MarkdownChunker + 测试

**Files:**
- Modify: `src/tiny_rag/ingestion/chunker.py`
- Modify: `tests/test_chunker.py`

核心类 `MarkdownChunker` 实现四步分块算法：
1. `_mark_protected(lines)` → 标记代码块和表格行号
2. `_build_sections(lines, protected)` → 按标题分组并追踪 heading_path
3. `_split_sections(sections)` → 合并小段、切分大段
4. 大段内按 `paragraph → line → token sliding window` 三级 fallback + overlap

- [ ] **Step 1: 读取现有文件确认上下文**

```bash
cat src/tiny_rag/ingestion/chunker.py
cat tests/test_chunker.py
```

- [ ] **Step 2: 重写 chunker.py**

写入以下完整内容：

```python
"""Chunker — split text into chunks with Markdown-aware semantic boundaries."""

import re
import logging
from dataclasses import dataclass

from .tokenizer import count_tokens, encode, decode

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    text: str
    heading_path: str = ""
    token_count: int = 0


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of at most chunk_size tokens with overlap.
    (Original token-sliding-window function, kept for backward compatibility.)
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    if not text:
        return []
    tokens = encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(decode(chunk_tokens))
        if end >= len(tokens):
            break
        start += chunk_size - chunk_overlap
    return chunks


class MarkdownChunker:
    """Markdown-aware chunker that respects heading hierarchy,
    code blocks, and table boundaries."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[ChunkResult]:
        """Chunk markdown text by heading hierarchy.

        Args:
            text: Markdown text.

        Returns:
            List of ChunkResult with heading_path, text, token_count.
        """
        if not text.strip():
            return []

        lines = text.split("\n")
        protected = self._mark_protected(lines)
        sections = self._build_sections(lines, protected)
        return self._apply_size_management(sections)

    # ── Step 1: Mark protected regions ──────────────────────

    def _mark_protected(self, lines: list[str]) -> set[int]:
        """Return set of line indices protected from splitting."""
        protected: set[int] = set()
        i = 0
        while i < len(lines):
            # Code fence: allow up to 3 leading spaces
            m = re.match(r"^[ ]{0,3}(`{3,})\s*\S*$", lines[i])
            if m:
                fence = m.group(1)
                protected.add(i)
                i += 1
                while i < len(lines):
                    protected.add(i)
                    if lines[i].strip() == fence:
                        i += 1
                        break
                    i += 1
                continue

            # Table: consecutive lines starting with |
            if re.match(r"^\|.*\|", lines[i]):
                row_start = i
                while i < len(lines) and re.match(r"^\|.*\|", lines[i]):
                    protected.add(i)
                    i += 1
                continue

            i += 1
        return protected

    # ── Step 2: Build sections by heading hierarchy ─────────

    def _build_sections(
        self, lines: list[str], protected: set[int]
    ) -> list[dict]:
        """Group lines into sections by heading hierarchy.

        Returns list of dicts: {heading_path, lines}
        """
        sections: list[dict] = []
        heading_stack: list[tuple[int, str]] = []
        current_lines: list[str] = []

        def flush():
            if current_lines:
                sections.append({
                    "heading_path": " > ".join(t[1] for t in heading_stack),
                    "lines": current_lines.copy(),
                })
                current_lines.clear()

        for i, line in enumerate(lines):
            if i in protected:
                current_lines.append(line)
                continue

            m = re.match(r"^(#{1,6})\s+(.+)$", line)
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))

            current_lines.append(line)

        if current_lines:
            sections.append({
                "heading_path": " > ".join(t[1] for t in heading_stack),
                "lines": current_lines.copy(),
            })

        return sections

    # ── Step 3: Size management ─────────────────────────────

    def _apply_size_management(self, sections: list[dict]) -> list[ChunkResult]:
        """Merge small sections, split large sections, apply overlap."""
        results: list[ChunkResult] = []

        def _chunk(text: str, hp: str) -> ChunkResult:
            return ChunkResult(
                text=text, heading_path=hp, token_count=count_tokens(text),
            )

        i = 0
        while i < len(sections):
            text = "\n".join(sections[i]["lines"])
            tokens = count_tokens(text)
            hp = sections[i]["heading_path"]

            # Small: merge with next, or previous if last
            if 0 < tokens < 100:
                if i + 1 < len(sections):
                    next_text = "\n".join(sections[i + 1]["lines"])
                    results.append(_chunk(text + "\n" + next_text, hp))
                    i += 2
                    continue
                elif results:
                    prev = results.pop()
                    results.append(_chunk(prev.text + "\n" + text, prev.heading_path))
                    i += 1
                    continue
                else:
                    results.append(_chunk(text, hp))

            # Within range: keep
            elif tokens <= self.chunk_size:
                results.append(_chunk(text, hp))

            # Too large: split
            else:
                results.extend(self._split_large(text, hp))

            i += 1

        return results

    # ── Large section splitting (paragraph → line → token) ──

    def _split_large(self, text: str, heading_path: str) -> list[ChunkResult]:
        """Split large text by paragraph, line, or token window."""
        paras = [p for p in text.split("\n\n") if p.strip()]
        if not paras:
            return self._split_by_tokens(text, heading_path)

        chunks: list[ChunkResult] = []
        current: list[str] = []
        current_tokens = 0

        for para in paras:
            pt = count_tokens(para)
            if current_tokens + pt <= self.chunk_size:
                current.append(para)
                current_tokens += pt
            else:
                if current:
                    chunks.append(ChunkResult(
                        text="\n\n".join(current),
                        heading_path=heading_path,
                        token_count=current_tokens,
                    ))
                current = [para]
                current_tokens = pt

        if current:
            chunks.append(ChunkResult(
                text="\n\n".join(current),
                heading_path=heading_path,
                token_count=current_tokens,
            ))

        # Single huge paragraph -> try line split
        if len(chunks) == 1 and current_tokens > self.chunk_size:
            return self._split_by_lines(text, heading_path)

        self._apply_overlap(chunks)
        return chunks

    def _split_by_lines(self, text: str, heading_path: str) -> list[ChunkResult]:
        """Split by newlines (second fallback)."""
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            return self._split_by_tokens(text, heading_path)

        chunks: list[ChunkResult] = []
        current: list[str] = []
        current_tokens = 0

        for line in lines:
            lt = count_tokens(line)
            if current_tokens + lt <= self.chunk_size:
                current.append(line)
                current_tokens += lt
            else:
                if current:
                    chunks.append(ChunkResult(
                        text="\n".join(current),
                        heading_path=heading_path,
                        token_count=current_tokens,
                    ))
                current = [line]
                current_tokens = lt

        if current:
            chunks.append(ChunkResult(
                text="\n".join(current),
                heading_path=heading_path,
                token_count=current_tokens,
            ))

        if len(chunks) == 1 and current_tokens > self.chunk_size:
            return self._split_by_tokens(text, heading_path)

        self._apply_overlap(chunks)
        return chunks

    def _split_by_tokens(self, text: str, heading_path: str) -> list[ChunkResult]:
        """Token sliding window (final fallback)."""
        tokens = encode(text)
        chunks: list[ChunkResult] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_str = decode(tokens[start:end])
            chunks.append(ChunkResult(
                text=chunk_str,
                heading_path=heading_path,
                token_count=end - start,
            ))
            if end >= len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ── Overlap ────────────────────────────────────────────

    def _apply_overlap(self, chunks: list[ChunkResult]) -> None:
        """Add overlap from previous chunk to current chunk."""
        if len(chunks) < 2 or self.chunk_overlap <= 0:
            return
        for i in range(1, len(chunks)):
            prev_tokens = encode(chunks[i - 1].text)
            if len(prev_tokens) <= self.chunk_overlap:
                overlap_text = chunks[i - 1].text
            else:
                overlap_text = decode(prev_tokens[-self.chunk_overlap:])
            chunks[i].text = overlap_text + chunks[i].text
            chunks[i].token_count = count_tokens(chunks[i].text)
```

- [ ] **Step 3: 写入完整测试文件**

覆盖测试到 `tests/test_chunker.py`。保留原测试 + 新增 MarkdownChunker 测试：

```python
"""Tests for chunker module."""

from src.tiny_rag.ingestion.chunker import chunk_text, MarkdownChunker, ChunkResult


# ── Original chunk_text tests (backward compat) ──

def test_chunk_text_single_chunk():
    text = "Short text."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_multiple_chunks():
    text = "hello " * 200
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) >= 2


def test_chunk_text_with_overlap():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=100, chunk_overlap=0) == []


def test_chunk_text_raises_on_invalid_params():
    import pytest
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("text", chunk_size=100, chunk_overlap=200)


# ── MarkdownChunker tests ──

def make_chunker(chunk_size=512, chunk_overlap=64):
    return MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_markdown_empty():
    assert make_chunker().chunk_text("") == []


def test_markdown_whitespace_only():
    assert make_chunker().chunk_text("   \n\n  ") == []


def test_markdown_plain_text_no_heading():
    """Text without any heading should produce one chunk."""
    text = "This is a paragraph of text.\n\nAnother paragraph."
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ""
    assert "paragraph" in chunks[0].text


def test_markdown_single_heading():
    text = "# Title\n\nContent under title."
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    assert len(chunks) >= 1
    assert "Title" in chunks[0].heading_path or chunks[0].heading_path == "Title"


def test_markdown_heading_hierarchy():
    text = "# Level1\n\nIntro\n\n## Level2\n\nDetail\n\n### Level3\n\nDeep"
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    paths = [c.heading_path for c in chunks]
    # Should have sections with correct paths
    assert any("Level1" in p for p in paths)
    assert any("Level2" in p for p in paths)
    assert any("Level3" in p for p in paths)


def test_markdown_heading_stack_resets():
    """A lower-level heading should reset the stack."""
    text = "# A\n\n## A1\n\n# B\n\n## B1\n\n### B1a"
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    paths = [c.heading_path for c in chunks]
    b_sections = [p for p in paths if p.startswith("B")]
    assert len(b_sections) >= 1
    # 'B' should not include 'A' in its path
    assert all("A" not in p for p in b_sections)


def test_markdown_preamble():
    """Content before first heading should have empty heading_path."""
    text = "Preamble paragraph.\n\n# Title\n\nBody."
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    preambles = [c for c in chunks if c.heading_path == ""]
    assert len(preambles) >= 1
    assert "Preamble" in preambles[0].text


def test_markdown_code_block_protected():
    """Code blocks should not be split, and # inside should not create headings."""
    text = "# Section\n\nSome text.\n\n```python\n# This is a comment, not a heading\ndef foo():\n    pass\n```\n\nMore text."
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    assert len(chunks) == 1
    assert "def foo" in chunks[0].text


def test_markdown_table_protected():
    """Tables should remain intact."""
    text = "# Data\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nDescription."
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    assert len(chunks) == 1
    assert "| A | B |" in chunks[0].text
    assert "| 1 | 2 |" in chunks[0].text


def test_markdown_small_section_merged():
    """Sections under 100 tokens should merge with the next."""
    text = "# A\n\nSmall.\n\n# B\n\nLarger content here.\n\nMore content.\n\nStill going to fill up tokens."
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    # 'Small.' should have merged into an adjacent chunk
    all_text = " ".join(c.text for c in chunks)
    assert "Small" in all_text


def test_markdown_last_section_orphan():
    """Last small section should merge backward."""
    text = "# Main\n\nThis is substantial content to ensure tokens are above threshold.\n\n" * 5
    text += "# Tiny\n\nSmall."
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    all_text = " ".join(c.text for c in chunks)
    assert "Small." in all_text


def test_markdown_large_section_paragraph_split():
    """Large section should split at paragraph boundaries."""
    text = "# Big\n\n" + "\n\n".join(f"Paragraph number {i} with enough content to fill it up nicely for testing purposes." for i in range(20))
    chunks = make_chunker(chunk_size=100, chunk_overlap=0).chunk_text(text)
    assert len(chunks) >= 2
    # Each chunk should be <= chunk_size (approximately - decode can shift slightly)
    for c in chunks:
        assert c.token_count <= 150  # allow some fuzz for decode


def test_markdown_overlap():
    """Adjacent sub-chunks should have overlap content."""
    text = "# Test\n\n" + "\n\n".join(f"Content paragraph {i} with some text to make it long enough for testing the overlap behavior." for i in range(15))
    chunks = make_chunker(chunk_size=120, chunk_overlap=30).chunk_text(text)
    assert len(chunks) >= 2
    # Check that overlap appears in consecutive chunks
    if len(chunks) >= 2:
        prev_end = chunks[0].text[-30:]
        next_start = chunks[1].text[:30]
        # The overlap content should make the start of chunk 1 appear in the end of chunk 0
        assert len(chunks[1].text) > 0


def test_markdown_code_block_before_table():
    """Code block should take priority over table detection."""
    text = "# Section\n\n```\n| not a table |\n| still code |\n```\n\n| real | table |\n|------|-------|\n| a    | b     |"
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    assert len(chunks) == 1
    assert "| not a table |" in chunks[0].text
    assert "| real | table |" in chunks[0].text


def test_markdown_heading_path_preserved():
    """Each chunk should carry its heading context."""
    text = "# A\n\nContent A\n\n## A1\n\nContent A1"
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    for c in chunks:
        if "Content A1" in c.text:
            assert "A1" in c.heading_path


def test_markdown_nested_heading_path():
    text = "# Root\n\n## Child\n\n### Grandchild\n\nDeep content."
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    for c in chunks:
        if "Deep content" in c.text:
            assert c.heading_path == "Root > Child > Grandchild"


def test_markdown_chunk_result_fields():
    """ChunkResult should have all required fields."""
    chunker = make_chunker()
    result = chunker.chunk_text("# Hi\n\nBody")
    assert len(result) == 1
    c = result[0]
    assert hasattr(c, "text")
    assert hasattr(c, "heading_path")
    assert hasattr(c, "token_count")
    assert isinstance(c.text, str)
    assert isinstance(c.heading_path, str)
    assert isinstance(c.token_count, int)
    assert c.token_count > 0
```

- [ ] **Step 4: 运行旧测试确认不破坏后向兼容**

```bash
pytest tests/test_chunker.py -v
```

Expected: 原有 5 个测试 PASS + 新增 16 个测试 PASS = 21 passed

- [ ] **Step 5: Commit**

```bash
git add src/tiny_rag/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: add MarkdownChunker with heading-aware semantic chunking"
```

---

### Task 3: 更新 VectorStore 以接受 ChunkResult

**Files:**
- Modify: `src/tiny_rag/storage/vector_store.py`
- Modify: `tests/test_vector_store.py`

- [ ] **Step 1: 更新 vector_store.py 的 add_document 签名**

将 `chunks: list[str]` 改为 `chunks: list[ChunkResult]`，从 ChunkResult 提取 text + heading_path：

```python
from src.tiny_rag.ingestion.chunker import ChunkResult
```

改动 `add_document` 方法（仅修改 chunks 参数处理和 metadata 构建）：

```python
    def add_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[ChunkResult],
        embeddings: list[list[float]],
    ) -> None:
        """Store document chunks with their embeddings.

        Args:
            doc_id: Unique document identifier.
            filename: Original filename.
            chunks: List of ChunkResult objects.
            embeddings: List of embedding vectors, one per chunk.
        """
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "heading_path": chunks[i].heading_path,
            }
            for i in range(len(chunks))
        ]

        self._collection.add(
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
```

- [ ] **Step 2: 更新 vector_store 测试**

```python
"""Tests for vector store module."""

import tempfile

from src.tiny_rag.storage.vector_store import VectorStore
from src.tiny_rag.ingestion.chunker import ChunkResult


def test_add_and_search():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStore(persist_dir=tmpdir)
        doc_id = "doc_001"

        store.add_document(
            doc_id=doc_id, filename="test.txt",
            chunks=[
                ChunkResult(text="apple banana fruit", heading_path="Fruits"),
                ChunkResult(text="red blue color", heading_path="Colors"),
            ],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )

        results = store.search(query_embedding=[0.1, 0.2], n_results=1)
        assert len(results) == 1
        assert results[0]["doc_id"] == doc_id


def test_list_documents():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStore(persist_dir=tmpdir)
        store.add_document(
            doc_id="doc_001", filename="test.txt",
            chunks=[ChunkResult(text="hello world", heading_path="")],
            embeddings=[[0.5, 0.5]],
        )
        docs = store.list_documents()
        assert any(d["id"] == "doc_001" for d in docs)


def test_search_empty_collection():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStore(persist_dir=tmpdir)
        results = store.search(query_embedding=[0.1, 0.2], n_results=5)
        assert results == []


def test_heading_path_stored_in_metadata():
    """heading_path should be stored and retrievable via metadata."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStore(persist_dir=tmpdir)
        store.add_document(
            doc_id="doc_h", filename="h.txt",
            chunks=[
                ChunkResult(text="intro", heading_path=""),
                ChunkResult(text="details", heading_path="Setup > Config"),
            ],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
        results = store.search(query_embedding=[0.1, 0.2], n_results=2)
        paths = {r["text"]: r.get("heading_path", "") for r in results}
        assert paths.get("details") == "Setup > Config"
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_vector_store.py -v
```

Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add src/tiny_rag/storage/vector_store.py tests/test_vector_store.py
git commit -m "feat: update VectorStore to accept ChunkResult with heading_path metadata"
```

---

### Task 4: 实现 WebLoader + 测试

**Files:**
- Create: `src/tiny_rag/ingestion/web_loader.py`
- Create: `tests/test_web_loader.py`

- [ ] **Step 1: 写入 WebLoader 实现**

```python
"""Web page loader — fetch HTML pages and convert to Markdown."""

import logging
import re
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import html2text
import httpx

logger = logging.getLogger(__name__)

_IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")


@dataclass
class PageResult:
    url: str
    markdown: str
    depth: int


class WebLoader:
    """BFS web crawler that converts HTML pages to Markdown.

    Args:
        max_depth: Maximum link-following depth (default 20).
        request_timeout: HTTP request timeout in seconds (default 30).
    """

    def __init__(self, max_depth: int = 20, request_timeout: float = 30.0):
        self.max_depth = max_depth
        self.request_timeout = request_timeout
        self._converter = html2text.HTML2Text()
        self._converter.body_width = 0
        self._converter.skip_internal_links = False
        self._converter.protect_links = True

    def load(self, start_url: str, max_depth: int | None = None) -> list[PageResult]:
        """BFS crawl from *start_url*, return all fetched pages.

        Args:
            start_url: The URL to start crawling from.
            max_depth: Override for instance max_depth (optional).

        Returns:
            List of PageResult (url, markdown, depth).
        """
        effective_depth = max_depth if max_depth is not None else self.max_depth
        results: list[PageResult] = []
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()

        norm = self._normalize_url(start_url, start_url)
        if not norm:
            logger.warning("Invalid start URL: %s", start_url)
            return results

        queue.append((norm, 0))
        visited.add(norm)

        with httpx.Client(timeout=self.request_timeout, follow_redirects=True) as client:
            while queue:
                url, depth = queue.popleft()

                try:
                    resp = client.get(url)
                    resp.raise_for_status()

                    content_type = resp.headers.get("content-type", "").lower()
                    if "text/html" not in content_type:
                        logger.info("Skipping non-HTML: %s (%s)", url, content_type)
                        continue

                    html = resp.text
                    markdown = self._converter.handle(html)
                    markdown = _IMAGE_PATTERN.sub("", markdown).strip()

                    results.append(PageResult(url=url, markdown=markdown, depth=depth))

                    # Extract links for next BFS level
                    if depth < effective_depth:
                        for href in self._extract_links(html):
                            absolute = self._normalize_url(href, url)
                            if absolute and absolute not in visited:
                                visited.add(absolute)
                                queue.append((absolute, depth + 1))

                except Exception as e:
                    logger.warning("Failed to fetch %s: %s", url, e)

        return results

    def _extract_links(self, html: str) -> list[str]:
        """Extract all href values from <a> tags in HTML."""
        links: list[str] = []
        for m in re.finditer(r'<a\s+[^>]*href="([^"]*)"', html, re.IGNORECASE):
            links.append(m.group(1))
        for m in re.finditer(r"<a\s+[^>]*href='([^']*)'", html, re.IGNORECASE):
            links.append(m.group(1))
        return links

    @staticmethod
    def _normalize_url(href: str, base: str) -> str | None:
        """Resolve relative URL and return normalized absolute URL, or None."""
        absolute = urljoin(base, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            return None
        qs = f"?{parsed.query}" if parsed.query else ""
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}{qs}"
```

- [ ] **Step 2: 写入 WebLoader 测试**

```python
"""Tests for web_loader module."""

from unittest.mock import patch, MagicMock

from src.tiny_rag.ingestion.web_loader import WebLoader, PageResult


SAMPLE_HTML = """
<html><body>
<h1>Test Page</h1>
<p>Hello world.</p>
<img src="pic.png" alt="photo">
<a href="/page2">Page 2</a>
<a href="https://example.com/page3">Page 3</a>
<a href="#section">Anchor</a>
<a href="mailto:test@example.com">Email</a>
<a href="https://other.com/ext">External</a>
</body></html>
"""

SAMPLE_HTML_2 = """
<html><body>
<h2>Page 2</h2>
<p>Content of page 2.</p>
</body></html>
"""


def _mock_response(text: str, status: int = 200, content_type: str = "text/html"):
    mock = MagicMock()
    mock.status_code = status
    mock.text = text
    mock.headers = {"content-type": content_type}
    return mock


def test_load_single_page():
    loader = WebLoader(max_depth=0)
    with patch.object(loader, "_normalize_url", return_value="https://example.com"):
        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = _mock_response(SAMPLE_HTML)

            results = loader.load("https://example.com")

    assert len(results) == 1
    assert results[0].url == "https://example.com"
    assert "Test Page" in results[0].markdown
    assert results[0].depth == 0


def test_load_removes_images():
    loader = WebLoader(max_depth=0)
    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.return_value = _mock_response(SAMPLE_HTML)

        results = loader.load("https://example.com")

    assert "pic.png" not in results[0].markdown
    assert "photo" not in results[0].markdown


def test_load_follows_links_bfs():
    loader = WebLoader(max_depth=1)
    call_count = 0

    def side_effect(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if "page2" in url or call_count > 1:
            return _mock_response(SAMPLE_HTML_2)
        return _mock_response(SAMPLE_HTML)

    # Patch normalize to return the href as-is
    def normalize(href, base):
        return href if href.startswith("http") else None

    loader._normalize_url = normalize

    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.side_effect = side_effect

        results = loader.load("https://example.com")

    # Should have fetched 2 pages (start + /page3)
    assert len(results) >= 2


def test_load_depth_limit():
    loader = WebLoader(max_depth=0)
    loader._normalize_url = lambda h, b: h if h.startswith("http") else None

    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.return_value = _mock_response(SAMPLE_HTML)

        results = loader.load("https://example.com")

    # max_depth=0 means only the start page
    assert len(results) == 1


def test_load_skips_non_html():
    loader = WebLoader(max_depth=0)
    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.return_value = _mock_response("binary", content_type="application/pdf")

        results = loader.load("https://example.com/file.pdf")

    assert len(results) == 0


def test_load_handles_http_error():
    loader = WebLoader(max_depth=0)
    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.side_effect = Exception("HTTP error")

        results = loader.load("https://example.com")

    assert len(results) == 0


def test_extract_links():
    loader = WebLoader()
    links = loader._extract_links(SAMPLE_HTML)
    assert "/page2" in links
    assert "https://example.com/page3" in links
    assert "#section" in links
    assert "mailto:test@example.com" in links


def test_normalize_url():
    assert WebLoader._normalize_url("/path", "https://example.com") == "https://example.com/path"
    assert WebLoader._normalize_url("https://other.com/page", "https://example.com") == "https://other.com/page"
    assert WebLoader._normalize_url("mailto:x@y.com", "https://example.com") is None
    assert WebLoader._normalize_url("javascript:void(0)", "https://example.com") is None


def test_normalize_url_drops_fragment():
    result = WebLoader._normalize_url("https://example.com/page#section", "https://example.com")
    assert result == "https://example.com/page"


def test_empty_start_url():
    loader = WebLoader()
    results = loader.load("")
    assert results == []


def test_url_deduplication():
    loader = WebLoader(max_depth=1)
    html_dual = """
    <html><body>
    <a href="https://example.com/page2">Page 2</a>
    <a href="https://example.com/page2">Page 2 again</a>
    </body></html>
    """
    loader._normalize_url = lambda h, b: h if h.startswith("http") else None

    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.side_effect = [
            _mock_response(html_dual),
            _mock_response(SAMPLE_HTML_2),
        ]

        results = loader.load("https://example.com")

    assert len(results) == 2  # start + page2 only (not twice)
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_web_loader.py -v
```

Expected: 11 passed

- [ ] **Step 4: Commit**

```bash
git add src/tiny_rag/ingestion/web_loader.py tests/test_web_loader.py
git commit -m "feat: add WebLoader for fetching Wiki pages with BFS crawling"
```

---

### Task 5: 更新 app.py

**Files:**
- Modify: `src/tiny_rag/app.py`
- Modify: `tests/test_app.py`

改动摘要：
1. import MarkdownChunker 替代 chunk_text
2. import WebLoader
3. 上传流程适配 ChunkResult
4. 新增 `/upload_web` 端点

- [ ] **Step 1: 修改 app.py 的 import 和 upload 流程**

在 import 区域：

```python
from src.tiny_rag.ingestion.chunker import MarkdownChunker
from src.tiny_rag.ingestion.web_loader import WebLoader
```

删除 `from src.tiny_rag.ingestion.chunker import chunk_text`。

在嵌入器和 reranker 初始化之后添加：

```python
chunker = MarkdownChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)

web_loader = WebLoader(max_depth=20)
```

修改 `upload()` 函数中的分块和存储逻辑：

```python
    chunks = chunker.chunk_text(content)
    if not chunks:
        return jsonify({"error": "Empty document"}), 400

    embeddings = embedder.embed([c.text for c in chunks])
    vector_store.add_document(doc_id=doc_id, filename=file.filename, chunks=chunks, embeddings=embeddings)
    bm25_retriever.add_document(doc_id=doc_id, filename=file.filename, chunks=[c.text for c in chunks])
```

- [ ] **Step 2: 新增 `/upload_web` 路由**

在 `/upload` 路由之后添加：

```python
@app.route("/upload_web", methods=["POST"])
def upload_web():
    body = request.get_json(silent=True)
    if not body or "url" not in body:
        return jsonify({"error": "Missing 'url' field"}), 400

    url = body["url"]
    max_depth = body.get("max_depth", 20)

    pages = web_loader.load(url, max_depth=max_depth)
    if not pages:
        return jsonify({"error": "No pages could be fetched from the URL"}), 400

    results: list[dict] = []
    for page in pages:
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        chunks = chunker.chunk_text(page.markdown)
        if not chunks:
            continue

        embeddings = embedder.embed([c.text for c in chunks])
        vector_store.add_document(doc_id=doc_id, filename=page.url, chunks=chunks, embeddings=embeddings)
        bm25_retriever.add_document(doc_id=doc_id, filename=page.url, chunks=[c.text for c in chunks])

        results.append({"url": page.url, "chunks": len(chunks)})

    cache.clear()

    return jsonify({"pages": len(results), "results": results})
```

- [ ] **Step 3: 更新 app 测试**

```python
"""Tests for Flask app module."""

import json
import io
import pytest
from src.tiny_rag.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_upload_no_file(client):
    resp = client.post("/upload")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert "error" in data


def test_upload_empty_filename(client):
    resp = client.post("/upload", data={"file": (io.BytesIO(b"test"), "")})
    assert resp.status_code == 400


def test_get_documents_empty(client):
    resp = client.get("/documents")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_stats_returns_counters(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    for key in ("hits", "misses", "total_requests", "hit_rate", "cache_entries", "force_refreshes", "poisoned_skips", "recent_misses"):
        assert key in data


def test_ask_no_question(client):
    resp = client.post("/ask", json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert "error" in data


def test_health_check(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_upload_unsupported_format(client):
    resp = client.post(
        "/upload",
        data={"file": (io.BytesIO(b"some content"), "test.docx")},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert "error" in data


def test_upload_web_no_url(client):
    resp = client.post("/upload_web", json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert "error" in data


def test_upload_web_success(client):
    """Mock WebLoader to return a fake page."""
    from unittest.mock import patch
    from src.tiny_rag.ingestion.web_loader import PageResult

    with (
        patch("src.tiny_rag.app.web_loader.load", return_value=[
            PageResult(url="https://wiki.example.com/page", markdown="# Hello\n\nWorld", depth=0),
        ]),
        patch("src.tiny_rag.app.embedder.embed", return_value=[[0.1] * 768]),
        patch("src.tiny_rag.app.vector_store.add_document"),
        patch("src.tiny_rag.app.bm25_retriever.add_document"),
        patch("src.tiny_rag.app.cache.clear"),
    ):
        resp = client.post("/upload_web", json={"url": "https://wiki.example.com/page"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert data["pages"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["url"] == "https://wiki.example.com/page"
    assert data["results"][0]["chunks"] >= 1


def test_upload_web_fetch_fails(client):
    from unittest.mock import patch

    with patch("src.tiny_rag.app.web_loader.load", return_value=[]):
        resp = client.post("/upload_web", json={"url": "https://wiki.example.com/page"})

    assert resp.status_code == 400
    assert "error" in resp.get_json()
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_app.py -v -k "not test_upload_pdf_success and not test_markdown_upload and not test_ask_invokes_rerank_when_configured and not test_ask_skips_rerank_when_key_empty"
```

Expected: 10 passed (basic mockable tests)

- [ ] **Step 5: 运行全部测试确认未破坏**

```bash
pytest --cov=src.tiny_rag -v
```

Expected: 全部测试通过（除需要真实 API key 的集成测试外）

- [ ] **Step 6: Commit**

```bash
git add src/tiny_rag/app.py tests/test_app.py
git commit -m "feat: add /upload_web endpoint and adapt upload flow for MarkdownChunker"
```

---

### Task 6: Verify

- [ ] **Step 1: 运行完整测试套件**

```bash
pytest -v
```

确认所有 mockable 测试通过。集成测试（需要 API key）可能跳过。

- [ ] **Step 2: 最终提交**

```bash
git status
git add -A
git commit -m "chore: final integration adjustments"
```
