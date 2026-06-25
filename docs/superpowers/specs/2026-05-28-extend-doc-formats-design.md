# 扩展文档格式支持 — 设计文档

**日期：** 2026-05-28
**项目：** Tiny RAG

## 概述

在现有 `.txt` 基础上增加 Markdown (`.md`) 和 PDF (`.pdf`) 文档格式支持。

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| PDF 解析库 | PyMuPDF (`fitz`) | 速度快、提取质量好、社区成熟 |
| Markdown 处理 | 直接读取纯文本，不剥离标记 | 简单、零依赖，Markdown 语法不影响检索效果 |

## 变更清单

### 1. 依赖

- `requirements.txt` 新增 `PyMuPDF`

### 2. `src/tiny_rag/ingestion/loader.py`

新增函数：

```python
def load_pdf(content: bytes) -> str:
    """用 PyMuPDF 提取 PDF 字节内容的全部文字。

    Args:
        content: PDF 文件的原始字节。

    Returns:
        提取的纯文本字符串。
    """
```

实现：`fitz.open(stream=content, filetype="pdf")` 遍历每页提取文字。

现有函数 `load_text()` 和 `load_bytes()` 保持不变，覆盖 `.txt` 和 `.md` 文件。

### 3. `src/tiny_rag/app.py`

- 定义 `ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}`
- `/upload` 路由增加扩展名校验
- 按扩展名分流：
  - `.pdf` → `load_pdf()`
  - `.txt` / `.md` → 现有 `load_bytes()`
- 不支持格式返回 400

### 4. `src/tiny_rag/config.py`

（无变更 — 文件扩展名校验直接在路由层处理，不增加配置项）

### 5. `src/tiny_rag/templates/index.html`

- `<input accept=".txt">` → `accept=".txt,.md,.pdf"`
- 上传区提示文字更新

### 6. 测试

#### `tests/test_loader.py`（新增）

- `test_load_pdf_extracts_text()` — 用小段合法 PDF 验证提取结果
- `test_load_markdown_as_text()` — .md 文件走 `load_text()` 验证（可选）

#### `tests/test_app.py`（扩展）

- `test_upload_pdf_success()` — 模拟上传 PDF 文件，验证 200
- `test_upload_unsupported_format()` — 上传 .docx 文件，验证 400

## 不涉及

- 输入格式类型不通过配置扩展（后续可通过列表配置，当前简单路由层处理即可）
- 不对 Markdown 做前置渲染或结构化解析
- 不修改向量存储、嵌入、生成的逻辑

## 边界情况

| 场景 | 行为 |
|------|------|
| 上传不含扩展名的文件 | 现有 `file.filename == ""` 校验已覆盖，返回 400 |
| 上传空 PDF | 提取空文本 → 分块返回空列表 → `chunks = []` 返回 400 |
| 上传加密 PDF | PyMuPDF 不支持无密码打开 → 异常返回 500（可后续优化）
| 超大 PDF | PyMuPDF 流式处理，无需临时文件 |
