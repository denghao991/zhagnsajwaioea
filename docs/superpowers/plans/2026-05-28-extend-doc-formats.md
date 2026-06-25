# 扩展文档格式支持 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 `.txt` 基础上增加 Markdown (`.md`) 和 PDF (`.pdf`) 文档上传支持。

**Architecture:** Markdown 直接读作 UTF-8 纯文本（零处理），PDF 用 PyMuPDF (`fitz`) 提取全部文字内容。上传路由按扩展名分流，现有分块/嵌入/检索/生成管线不变。

**Tech Stack:** PyMuPDF

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `requirements.txt` | 修改 | 添加 `PyMuPDF` |
| `src/tiny_rag/ingestion/loader.py` | 修改 | 新增 `load_pdf()` |
| `src/tiny_rag/app.py` | 修改 | 扩展名校验 + PDF 分流 |
| `src/tiny_rag/templates/index.html` | 修改 | accept 属性和提示文字 |
| `tests/test_loader.py` | 新建 | PDF loader 和 Markdown 读取测试 |
| `tests/test_app.py` | 修改 | 增加 PDF 上传和格式拒绝测试 |

---

### Task 1: 添加 PyMuPDF 依赖

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 在 requirements.txt 中添加 PyMuPDF**

在文件末尾添加：
```
pymupdf==1.25.5
```

- [ ] **Step 2: 安装并验证**

Run: `pip install pymupdf`

Expected: 安装成功，无报错。

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add PyMuPDF dependency"
```

---

### Task 2: Loader 模块 — 新增 load_pdf()

**Files:**
- Modify: `src/tiny_rag/ingestion/loader.py`
- Create: `tests/test_loader.py`

- [ ] **Step 1: 编写失败的测试** (`tests/test_loader.py`)

```python
"""Tests for document loader module."""

from pathlib import Path

from src.tiny_rag.ingestion.loader import load_text, load_bytes, load_pdf


def test_load_pdf_extracts_text():
    """A minimal valid PDF containing 'Hello PDF'."""
    # Minimal PDF: header + object with stream
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"5 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET\n"
        b"endstream\n"
        b"endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000348 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n"
        b"452\n"
        b"%%EOF"
    )
    text = load_pdf(pdf_bytes)
    assert isinstance(text, str)
    assert len(text) > 0


def test_load_text_markdown(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\n\nHello **world**.", encoding="utf-8")
    content = load_text(str(md_file))
    assert "# Title" in content
    assert "**world**" in content
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_loader.py -v --tb=short`

Expected: `ImportError: cannot import name 'load_pdf'` （`loader.py` 中尚无此函数）

- [ ] **Step 3: 在 `loader.py` 中实现 `load_pdf()`**

在 `load_bytes()` 之后追加：

```python
def load_pdf(content: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF.

    Args:
        content: Raw bytes of a PDF file.

    Returns:
        Extracted plain text.
    """
    import fitz

    doc = fitz.open(stream=content, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)
```

同时更新文件顶部的文档字符串（可选），无需修改现有函数。

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_loader.py -v --tb=short`

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/tiny_rag/ingestion/loader.py tests/test_loader.py
git commit -m "feat: add PDF text extraction via PyMuPDF"
```

---

### Task 3: Flask 路由 — 扩展名检查和 PDF 分流

**Files:**
- Modify: `src/tiny_rag/app.py`

- [ ] **Step 1: 在 `app.py` 中增加扩展名校验和 PDF 分流**

在文件顶部 import 区，追加 `Path`（已有）和 `load_pdf` 的导入：

在 `from src.tiny_rag.ingestion.loader import load_bytes` 行，改为：

```python
from src.tiny_rag.ingestion.loader import load_bytes, load_pdf
```

在 `import uuid` 之后（或在 `import` 块末尾），增加允许的扩展名集合：

```python
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
```

修改 `/upload` 路由函数，在 `file.filename == ""` 校验之后增加：

```python
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "不支持的文件格式，仅支持 .txt / .md / .pdf"}), 400

    raw = file.read()

    if ext == ".pdf":
        content = load_pdf(raw)
    else:
        content = load_bytes(raw)
```

并将原有的 `content = load_bytes(file.read())` 一行移除。

最终 `/upload` 函数的完整代码应为：

```python
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "不支持的文件格式，仅支持 .txt / .md / .pdf"}), 400

    raw = file.read()
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"

    if ext == ".pdf":
        content = load_pdf(raw)
    else:
        content = load_bytes(raw)

    chunks = chunk_text(
        content,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        return jsonify({"error": "Empty document"}), 400

    embeddings = embedder.embed(chunks)
    vector_store.add_document(doc_id=doc_id, chunks=chunks, embeddings=embeddings)

    return jsonify({"id": doc_id, "filename": file.filename, "chunks": len(chunks)})
```

- [ ] **Step 2: 运行现有的纯逻辑测试验证不破坏已有功能**

Run: `python -m pytest tests/test_app.py -v --tb=short`

Expected: 5 tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/tiny_rag/app.py
git commit -m "feat: add file extension validation and PDF upload support"
```

---

### Task 4: 更新前端 — 接受 PDF 和 Markdown 文件

**Files:**
- Modify: `src/tiny_rag/templates/index.html`

- [ ] **Step 1: 修改文件输入 accept 属性和提示文字**

将 `<input type="file" id="file-input" accept=".txt">` 改为：

```html
      <input type="file" id="file-input" accept=".txt,.md,.pdf">
```

将上传区域的提示文字 `<div class="hint" id="upload-status">支持纯文本文件</div>` 改为：

```html
      <div class="hint" id="upload-status">支持 .txt / .md / .pdf 文件</div>
```

- [ ] **Step 2: 手动验证（可选，确认模板无渲染错误）**

Run: `python -c "from src.tiny_rag.app import app; print('OK')"`

Expected: 打印 "OK"

- [ ] **Step 3: Commit**

```bash
git add src/tiny_rag/templates/index.html
git commit -m "feat: update frontend to accept .md and .pdf files"
```

---

### Task 5: 扩展测试 — 增加 PDF 上传和格式拒绝测试

**Files:**
- Modify: `tests/test_app.py`

- [ ] **Step 1: 在 `test_app.py` 中增加 PDF 上传和非法格式测试**

在文件末尾追加：

```python
def test_upload_pdf_success(client):
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"5 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET\n"
        b"endstream\n"
        b"endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000348 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n"
        b"452\n"
        b"%%EOF"
    )
    resp = client.post(
        "/upload",
        data={"file": (io.BytesIO(pdf_bytes), "test.pdf")},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert "id" in data


def test_upload_unsupported_format(client):
    resp = client.post(
        "/upload",
        data={"file": (io.BytesIO(b"some content"), "test.docx")},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert "error" in data


def test_markdown_upload(client):
    md_content = b"# Title\n\nHello **world**."
    resp = client.post(
        "/upload",
        data={"file": (io.BytesIO(md_content), "test.md")},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert "id" in data
```

注意：`test_upload_pdf_success` 和 `test_markdown_upload` 实际上会调用 `embedder.embed()`，如果没有 API key 会失败。仅测试路由校验逻辑时，可以加 `@pytest.mark.skipif` 跳过它们。但如果已经有 API key，它们会真正运行完整的上传流程。

为了不依赖外部 API，让这些测试只验证路由层的逻辑，需要在 `monkeypatch` 中 mock 掉 `embedder.embed` 和 `vector_store.add_document`：

实际上，当前 `test_app.py` 的 fixture 已经配置了 `TESTING` 模式。更简单的做法：仅测试 `test_upload_unsupported_format` 这不依赖外部服务的用例。PDF 和 Markdown 的成功上传测试将需要 mock 或是真实 API key。我们暂时只保留 `test_upload_unsupported_format` 这个确定能通过的测试，另外两个加上 `@pytest.mark.skip(reason="需要 API key")` 以便未来手动验证。

改为：

```python
import pytest


def test_upload_pdf_success(client):
    """Requires valid API keys in .env to pass."""
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"5 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET\n"
        b"endstream\n"
        b"endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000348 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n"
        b"452\n"
        b"%%EOF"
    )
    resp = client.post(
        "/upload",
        data={"file": (io.BytesIO(pdf_bytes), "test.pdf")},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert "id" in data


def test_upload_unsupported_format(client):
    resp = client.post(
        "/upload",
        data={"file": (io.BytesIO(b"some content"), "test.docx")},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert "error" in data


def test_markdown_upload(client):
    """Requires valid API keys in .env to pass."""
    md_content = b"# Title\n\nHello **world**."
    resp = client.post(
        "/upload",
        data={"file": (io.BytesIO(md_content), "test.md")},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert "id" in data
```

- [ ] **Step 2: 运行测试验证通过**

Run: `python -m pytest tests/test_app.py -v --tb=short`

Expected: 
- `test_upload_no_file` PASS
- `test_upload_empty_filename` PASS
- `test_get_documents_empty` PASS
- `test_ask_no_question` PASS
- `test_health_check` PASS
- `test_upload_pdf_success` PASS（需要有效 API key，已在 `.env` 中配置）
- `test_upload_unsupported_format` PASS
- `test_markdown_upload` PASS（需要有效 API key，已在 `.env` 中配置）

- [ ] **Step 3: Commit**

```bash
git add tests/test_app.py
git commit -m "test: add PDF upload and unsupported format tests"
```

---

### Task 6: 完整回归测试

**Files:** （无新文件 — 运行全部测试）

- [ ] **Step 1: 运行全部测试**

Run: `python -m pytest tests/ -v --tb=short`

Expected: 所有 tests PASS（需要有效 API key 在 `.env` 中）

- [ ] **Step 2: 最终 commit**

```bash
git add -A
git commit -m "chore: finalize document format extension"
```
