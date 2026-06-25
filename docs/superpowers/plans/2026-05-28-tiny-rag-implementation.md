# Tiny RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a minimal RAG (Retrieval-Augmented Generation) system with Flask, ChromaDB, and OpenAI-compatible APIs.

**Architecture:** Upload → chunk (tiktoken) → embed (Qwen) → store (ChromaDB). Ask → embed query → retrieve chunks → generate answer (GLM). All via OpenAI-compatible API SDK.

**Tech Stack:** Flask 3.1, ChromaDB 1.5, OpenAI SDK 1.109, tiktoken 0.13, pydantic-settings 2.14

---

## File Structure

```
tiny-rag/
├── src/tiny_rag/
│   ├── __init__.py              # Package marker
│   ├── config.py                # pydantic-settings: API keys, model names, chunk params
│   ├── app.py                   # Flask entry point: /upload, /ask, /documents
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── tokenizer.py         # tiktoken wrapper (count, encode, decode)
│   │   ├── chunker.py           # Fixed-token chunking with overlap
│   │   └── loader.py            # Plain text file loader
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── client.py            # Qwen text-embedding-v2 via OpenAI SDK
│   ├── storage/
│   │   ├── __init__.py
│   │   └── vector_store.py      # ChromaDB PersistentClient wrapper
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm.py               # GLM chat via OpenAI SDK
│   └── templates/
│       └── index.html            # Single-page Web UI
├── tests/
│   ├── __init__.py
│   ├── test_tokenizer.py
│   ├── test_chunker.py
│   ├── test_vector_store.py
│   └── test_app.py
```

---

### Task 1: Directory Scaffolding

**Files:**
- Create: `src/tiny_rag/__init__.py`
- Create: `src/tiny_rag/ingestion/__init__.py`
- Create: `src/tiny_rag/embedding/__init__.py`
- Create: `src/tiny_rag/storage/__init__.py`
- Create: `src/tiny_rag/generation/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create all package init files**

Create empty `__init__.py` files for all packages:
```bash
mkdir -p src/tiny_rag/ingestion src/tiny_rag/embedding src/tiny_rag/storage src/tiny_rag/generation src/tiny_rag/templates tests
touch src/tiny_rag/__init__.py src/tiny_rag/ingestion/__init__.py src/tiny_rag/embedding/__init__.py src/tiny_rag/storage/__init__.py src/tiny_rag/generation/__init__.py tests/__init__.py
```

- [ ] **Step 2: Verify structure**

Run: `find src/tests -name "*.py" | sort` (or `dir /s /b src tests` on Windows)

Expected: all 6 `__init__.py` files present.

- [ ] **Step 3: Commit**

```bash
git add src/ tests/
git commit -m "chore: scaffold project directory structure"
```

---

### Task 2: Config Module

**Files:**
- Create: `src/tiny_rag/config.py`
- Test: (manual verification via import)

- [ ] **Step 1: Write `config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # GLM (LLM)
    llm_api_key: str = ""
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_model: str = "glm-4-plus"

    # DashScope / Qwen (Embedding)
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v2"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64


settings = Settings()
```

- [ ] **Step 2: Verify import**

Run: `cd <project_root> && python -c "from src.tiny_rag.config import settings; print(settings.model_dump())"`

Expected: prints all settings with defaults (no crash).

- [ ] **Step 3: Commit**

```bash
git add src/tiny_rag/config.py
git commit -m "feat: add pydantic-settings config module"
```

---

### Task 3: Tokenizer Module

**Files:**
- Create: `src/tiny_rag/ingestion/tokenizer.py`
- Test: `tests/test_tokenizer.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for tokenizer module."""

from src.tiny_rag.ingestion.tokenizer import count_tokens, encode, decode


def test_count_tokens_returns_positive_int():
    result = count_tokens("Hello, world!")
    assert isinstance(result, int)
    assert result > 0


def test_encode_decode_roundtrip():
    text = "The quick brown fox jumps over the lazy dog."
    tokens = encode(text)
    decoded = decode(tokens)
    assert decoded == text


def test_count_tokens_empty_string():
    assert count_tokens("") == 0


def test_encode_empty_string():
    assert encode("") == []


def test_decode_empty_list():
    assert decode([]) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd <project_root> && python -m pytest tests/test_tokenizer.py -v`

Expected: ModuleNotFoundError or ImportError (tokenizer.py doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
"""Tokenizer — tiktoken wrapper for text encoding/decoding."""

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return number of tokens in text."""
    return len(_ENCODING.encode(text, disallowed_special=()))


def encode(text: str) -> list[int]:
    """Encode text to token IDs."""
    return _ENCODING.encode(text, disallowed_special=())


def decode(tokens: list[int]) -> str:
    """Decode token IDs back to text."""
    return _ENCODING.decode(tokens)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd <project_root> && python -m pytest tests/test_tokenizer.py -v`

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiny_rag/ingestion/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: add tiktoken tokenizer wrapper"
```

---

### Task 4: Chunker Module

**Files:**
- Create: `src/tiny_rag/ingestion/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for chunker module."""

from src.tiny_rag.ingestion.chunker import chunk_text


def test_chunk_text_single_chunk():
    text = "Short text."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_multiple_chunks():
    text = "hello " * 200  # ~1000 tokens
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) >= 2


def test_chunk_text_with_overlap():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    # Verify no chunk is empty
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=100, chunk_overlap=0) == []


def test_chunk_text_raises_on_invalid_params():
    import pytest
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("text", chunk_size=100, chunk_overlap=200)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd <project_root> && python -m pytest tests/test_chunker.py -v`

Expected: ImportError (chunker.py doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
"""Chunker — split text into fixed-token chunks with overlap."""

from src.tiny_rag.ingestion.tokenizer import encode, decode, count_tokens


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of at most chunk_size tokens with overlap.

    Args:
        text: Input text to split.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Number of overlapping tokens between adjacent chunks.

    Returns:
        List of text chunks.

    Raises:
        ValueError: If chunk_overlap >= chunk_size.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd <project_root> && python -m pytest tests/test_chunker.py -v`

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiny_rag/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: add text chunker with overlap support"
```

---

### Task 5: Document Loader

**Files:**
- Create: `src/tiny_rag/ingestion/loader.py`
- Test: (covered by integration tests in app)

- [ ] **Step 1: Write `loader.py`**

```python
"""Document loader — reads text files from disk or bytes."""

from pathlib import Path


def load_text(file_path: str | Path) -> str:
    """Read text content from a file path.

    Args:
        file_path: Path to a .txt file.

    Returns:
        File contents as a string.
    """
    return Path(file_path).read_text(encoding="utf-8")


def load_bytes(content: bytes) -> str:
    """Decode raw bytes to UTF-8 text.

    Args:
        content: Raw bytes from uploaded file.

    Returns:
        Decoded string.
    """
    return content.decode("utf-8")
```

- [ ] **Step 2: Verify import**

Run: `cd <project_root> && python -c "from src.tiny_rag.ingestion.loader import load_text, load_bytes; print('OK')"`

Expected: prints "OK".

- [ ] **Step 3: Commit**

```bash
git add src/tiny_rag/ingestion/loader.py
git commit -m "feat: add document loader for plain text"
```

---

### Task 6: Embedding Client

**Files:**
- Create: `src/tiny_rag/embedding/client.py`
- Test: (manual — requires DashScope API key)

- [ ] **Step 1: Write `client.py`**

```python
"""Embedding client — Qwen text-embedding-v2 via OpenAI SDK."""

from openai import OpenAI


class EmbeddingClient:
    """Generate embeddings using OpenAI-compatible embedding API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [item.embedding for item in response.data]
```

- [ ] **Step 2: Verify import**

Run: `cd <project_root> && python -c "from src.tiny_rag.embedding.client import EmbeddingClient; print('OK')"`

Expected: prints "OK".

- [ ] **Step 3: Commit**

```bash
git add src/tiny_rag/embedding/client.py
git commit -m "feat: add Qwen embedding client via OpenAI SDK"
```

---

### Task 7: Vector Store

**Files:**
- Create: `src/tiny_rag/storage/vector_store.py`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for vector store module."""

import tempfile
from pathlib import Path

from src.tiny_rag.storage.vector_store import VectorStore


def test_add_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_dir=tmpdir)
        doc_id = "doc_001"

        store.add_document(
            doc_id=doc_id,
            chunks=["apple banana fruit", "red blue color"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )

        results = store.search(query_embedding=[0.1, 0.2], n_results=1)
        assert len(results) == 1
        assert results[0]["doc_id"] == doc_id


def test_list_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_dir=tmpdir)
        store.add_document(
            doc_id="doc_001",
            chunks=["hello world"],
            embeddings=[[0.5, 0.5]],
        )
        docs = store.list_documents()
        assert any(d["id"] == "doc_001" for d in docs)


def test_search_empty_collection():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_dir=tmpdir)
        results = store.search(query_embedding=[0.1, 0.2], n_results=5)
        assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd <project_root> && python -m pytest tests/test_vector_store.py -v`

Expected: ImportError (vector_store.py doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
"""Vector store — ChromaDB wrapper for document storage and retrieval."""

from collections.abc import Sequence

import chromadb
from chromadb.config import Settings


class VectorStore:
    """ChromaDB-based vector store for RAG document storage."""

    def __init__(self, persist_dir: str) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="documents",
        )

    def add_document(
        self,
        doc_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Store document chunks with their embeddings.

        Args:
            doc_id: Unique document identifier.
            chunks: List of text chunks.
            embeddings: List of embedding vectors, one per chunk.
        """
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

        self._collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict]:
        """Search for most similar chunks by embedding.

        Args:
            query_embedding: Embedding vector of the query.
            n_results: Number of top results to return.

        Returns:
            List of result dicts with keys: doc_id, chunk_index, text, distance.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        for i in range(len(results["ids"][0])):
            output.append({
                "doc_id": results["metadatas"][0][i].get("doc_id", ""),
                "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def list_documents(self) -> list[dict]:
        """List all unique documents and their chunk counts.

        Returns:
            List of dicts with keys: id, chunks_count.
        """
        all_meta = self._collection.get(include=["metadatas"])
        doc_map: dict[str, int] = {}
        for meta in all_meta["metadatas"]:
            doc_id = meta["doc_id"]
            doc_map[doc_id] = doc_map.get(doc_id, 0) + 1

        return [
            {"id": doc_id, "chunks": count}
            for doc_id, count in sorted(doc_map.items())
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd <project_root> && python -m pytest tests/test_vector_store.py -v`

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tiny_rag/storage/vector_store.py tests/test_vector_store.py
git commit -m "feat: add ChromaDB vector store wrapper"
```

---

### Task 8: LLM Client

**Files:**
- Create: `src/tiny_rag/generation/llm.py`
- Test: (manual — requires GLM API key)

- [ ] **Step 1: Write `llm.py`**

```python
"""LLM client — GLM chat via OpenAI SDK."""

from openai import OpenAI

_SYSTEM_PROMPT = """你是一个基于文档内容回答问题的助手。
请根据以下提供的文档片段来回应用户的问题。
如果你在文档中找不到相关信息，请诚实地说明你不知道。
不要编造信息。"""


class LLMClient:
    """Generate answers using OpenAI-compatible chat API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def generate(self, question: str, context: str) -> str:
        """Generate an answer based on question and retrieved context.

        Args:
            question: User's question.
            context: Retrieved document chunks joined as context.

        Returns:
            Generated answer text.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"文档内容：\n{context}\n\n问题：{question}"},
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""
```

- [ ] **Step 2: Verify import**

Run: `cd <project_root> && python -c "from src.tiny_rag.generation.llm import LLMClient; print('OK')"`

Expected: prints "OK".

- [ ] **Step 3: Commit**

```bash
git add src/tiny_rag/generation/llm.py
git commit -m "feat: add GLM LLM client via OpenAI SDK"
```

---

### Task 9: Flask App

**Files:**
- Create: `src/tiny_rag/app.py`
- Test: `tests/test_app.py`

- [ ] **Step 1: Write the failing test**

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
    assert data == {"documents": []}


def test_ask_no_question(client):
    resp = client.post("/ask", json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert "error" in data


def test_health_check(client):
    resp = client.get("/")
    assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd <project_root> && python -m pytest tests/test_app.py -v`

Expected: ImportError (app.py doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
"""Flask application — RAG system web server."""

import uuid
from pathlib import Path

from flask import Flask, jsonify, request, render_template

from src.tiny_rag.config import settings
from src.tiny_rag.ingestion.loader import load_bytes
from src.tiny_rag.ingestion.chunker import chunk_text
from src.tiny_rag.embedding.client import EmbeddingClient
from src.tiny_rag.storage.vector_store import VectorStore
from src.tiny_rag.generation.llm import LLMClient

app = Flask(__name__)

embedder = EmbeddingClient(
    base_url=settings.dashscope_base_url,
    api_key=settings.dashscope_api_key,
    model=settings.embedding_model,
)

vector_store = VectorStore(persist_dir=settings.chroma_persist_dir)

llm = LLMClient(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    model=settings.llm_model,
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    content = load_bytes(file.read())
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"

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


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True)
    if not body or "question" not in body:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = body["question"]
    question_embedding = embedder.embed([question])[0]
    results = vector_store.search(question_embedding, n_results=5)

    if not results:
        return jsonify({"answer": "未找到相关文档，请先上传文档。", "sources": []})

    context = "\n\n".join(r["text"] for r in results)
    answer = llm.generate(question=question, context=context)
    source_ids = list({r["doc_id"] for r in results})

    return jsonify({"answer": answer, "sources": source_ids})


@app.route("/documents", methods=["GET"])
def documents():
    return jsonify({"documents": vector_store.list_documents()})
```

- [ ] **Step 4: Run the logic-free tests to verify they pass**

Note: The tests in Step 1 test pure HTTP logic (no embedding/LLM calls needed). Run:

Run: `cd <project_root> && python -m pytest tests/test_app.py -v`

Expected: all 5 tests PASS (no API keys required for these test cases).

- [ ] **Step 5: Commit**

```bash
git add src/tiny_rag/app.py tests/test_app.py
git commit -m "feat: add Flask app with /upload, /ask, /documents routes"
```

---

### Task 10: Web Template

**Files:**
- Create: `src/tiny_rag/templates/index.html`

- [ ] **Step 1: Write `index.html`**

A clean single-page interface with:
- File upload form with drag-and-drop or file picker
- Chat-style question input and answer display
- Document list sidebar

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tiny RAG</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; min-height: 100vh; }
  .container { display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }
  .sidebar { background: #fff; border-right: 1px solid #e0e0e0; padding: 20px; }
  .sidebar h2 { font-size: 14px; text-transform: uppercase; color: #666; margin-bottom: 12px; }
  .sidebar ul { list-style: none; }
  .sidebar li { padding: 8px 12px; background: #f8f8f8; border-radius: 6px; margin-bottom: 6px; font-size: 13px; }
  .sidebar li small { color: #999; display: block; margin-top: 2px; }
  .main { display: flex; flex-direction: column; padding: 24px; max-width: 800px; margin: 0 auto; width: 100%; }
  .upload-area { background: #fff; border: 2px dashed #d0d0d0; border-radius: 12px; padding: 32px; text-align: center; margin-bottom: 24px; cursor: pointer; transition: border-color .2s; }
  .upload-area:hover { border-color: #4a90d9; }
  .upload-area input { display: none; }
  .upload-area p { color: #666; font-size: 14px; }
  .upload-area .hint { font-size: 12px; color: #999; margin-top: 8px; }
  .chat-area { flex: 1; display: flex; flex-direction: column; }
  .messages { flex: 1; overflow-y: auto; margin-bottom: 16px; }
  .message { padding: 12px 16px; border-radius: 12px; margin-bottom: 12px; max-width: 85%; line-height: 1.6; font-size: 14px; }
  .message.user { background: #4a90d9; color: #fff; margin-left: auto; }
  .message.assistant { background: #fff; border: 1px solid #e0e0e0; }
  .message .sources { font-size: 12px; color: #999; margin-top: 8px; }
  .input-row { display: flex; gap: 8px; }
  .input-row input { flex: 1; padding: 12px 16px; border: 1px solid #d0d0d0; border-radius: 8px; font-size: 14px; outline: none; }
  .input-row input:focus { border-color: #4a90d9; }
  .input-row button { padding: 12px 24px; background: #4a90d9; color: #fff; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; }
  .input-row button:disabled { background: #ccc; cursor: not-allowed; }
  .toast { position: fixed; top: 20px; right: 20px; padding: 12px 20px; border-radius: 8px; color: #fff; font-size: 14px; display: none; }
  .toast.success { background: #52c41a; }
  .toast.error { background: #ff4d4f; }
</style>
</head>
<body>
<div class="container">
  <aside class="sidebar">
    <h2>📄 文档列表</h2>
    <ul id="doc-list"><li style="color:#999">暂无文档</li></ul>
  </aside>
  <div class="main">
    <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click()">
      <input type="file" id="file-input" accept=".txt">
      <p>点击上传 .txt 文档</p>
      <div class="hint" id="upload-status">支持纯文本文件</div>
    </div>
    <div class="chat-area">
      <div class="messages" id="messages">
        <div class="message assistant">你好！请上传文档，然后向我提问。</div>
      </div>
      <div class="input-row">
        <input type="text" id="question-input" placeholder="输入你的问题..." onkeydown="if(event.key==='Enter') ask()">
        <button id="ask-btn" onclick="ask()">发送</button>
      </div>
    </div>
  </div>
</div>
<div id="toast" class="toast"></div>

<script>
document.getElementById('file-input').addEventListener('change', async function(e) {
  const file = e.target.files[0];
  if (!file) return;
  const status = document.getElementById('upload-status');
  status.textContent = '上传中...';
  const formData = new FormData();
  formData.append('file', file);
  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const data = await resp.json();
    if (resp.ok) {
      showToast(`上传成功：${data.chunks} 个分块`, 'success');
      status.textContent = `已上传 ${file.name}`;
      loadDocuments();
    } else {
      showToast(data.error, 'error');
      status.textContent = '上传失败';
    }
  } catch {
    showToast('网络错误', 'error');
    status.textContent = '上传失败';
  }
  e.target.value = '';
});

async function ask() {
  const input = document.getElementById('question-input');
  const btn = document.getElementById('ask-btn');
  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  btn.disabled = true;
  addMessage(question, 'user');
  addMessage('思考中...', 'assistant', 'thinking');
  try {
    const resp = await fetch('/ask', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question}) });
    const data = await resp.json();
    document.querySelector('.thinking')?.remove();
    if (resp.ok) {
      let html = data.answer;
      if (data.sources && data.sources.length) html += '<div class="sources">来源: ' + data.sources.join(', ') + '</div>';
      addMessage(html, 'assistant');
    } else {
      addMessage('错误: ' + data.error, 'assistant');
    }
  } catch {
    document.querySelector('.thinking')?.remove();
    addMessage('网络错误', 'assistant');
  }
  btn.disabled = false;
}

function addMessage(text, role, cls='') {
  const div = document.createElement('div');
  div.className = 'message ' + role + (cls ? ' ' + cls : '');
  div.innerHTML = text;
  document.getElementById('messages').appendChild(div);
  div.scrollIntoView({behavior: 'smooth'});
}

async function loadDocuments() {
  try {
    const resp = await fetch('/documents');
    const data = await resp.json();
    const list = document.getElementById('doc-list');
    if (data.documents.length === 0) {
      list.innerHTML = '<li style="color:#999">暂无文档</li>';
    } else {
      list.innerHTML = data.documents.map(d => `<li>${d.id}<small>${d.chunks} 个分块</small></li>`).join('');
    }
  } catch {}
}

function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = 'toast ' + type; t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 3000);
}

loadDocuments();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify template renders**

Run: `cd <project_root> && python -c "from flask import render_template; from src.tiny_rag.app import app; print('OK')"`

Expected: prints "OK".

- [ ] **Step 3: Commit**

```bash
git add src/tiny_rag/templates/index.html
git commit -m "feat: add web UI template"
```

---

### Task 11: Verify Integration

**Files:** (no new files — run the full app)

- [ ] **Step 1: Start the server briefly to check it boots**

Run: `cd <project_root> && timeout 5 python -m src.tiny_rag.app 2>&1 || true`

Expected: Flask starts on port 5000 (no import errors).

Note: Full integration test (upload + ask) requires valid API keys in `.env`.

- [ ] **Step 2: Run all tests**

Run: `cd <project_root> && python -m pytest tests/ -v --tb=short`

Expected: all tests PASS.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: finalize initial implementation"
```
