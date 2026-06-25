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
    for key in ("hits", "misses", "total_requests", "hit_rate", "cache_entries", "threshold", "max_entries"):
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
        data={"file": (io.BytesIO(b"some content"), "test.txt")},
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


def test_ask_invokes_rerank_when_configured(client):
    """RerankClient.rerank is called in /ask when rerank_llm_api_key is set."""
    from unittest.mock import ANY, patch, Mock
    from src.tiny_rag.config import settings

    mock_docs = [
        {"text": "test chunk", "doc_id": "doc1", "filename": "test.md", "chunk_index": 0, "score": 0.9},
    ]

    original_key = settings.rerank_llm_api_key
    settings.rerank_llm_api_key = "sk-test"
    try:
        with (
            patch("src.tiny_rag.app.embedder.embed", return_value=[[0.1] * 768]),
            patch("src.tiny_rag.app.vector_store.search", return_value=mock_docs),
            patch("src.tiny_rag.app.bm25_retriever.search", return_value=[]),
            patch("src.tiny_rag.app.llm.rewrite", return_value="test query"),
            patch("src.tiny_rag.app.llm.generate_stream", return_value=iter(["answer"])),
            patch("src.tiny_rag.app.reranker.rerank", return_value=mock_docs) as mock_rerank,
        ):
            resp = client.post("/ask", json={"question": "test query"})
    finally:
        settings.rerank_llm_api_key = original_key

    assert resp.status_code == 200
    mock_rerank.assert_called_once()
    call_args = mock_rerank.call_args
    assert call_args[0][0] == "test query"  # query (positional)
    assert call_args[0][1] == mock_docs     # documents (positional)
    assert call_args[1] == {"top_n": 5}     # top_n (keyword)


def test_ask_skips_rerank_when_key_empty(client):
    """Rerank is not called when rerank_llm_api_key is empty."""
    from unittest.mock import patch
    from src.tiny_rag.config import settings

    mock_docs = [
        {"text": "test chunk", "doc_id": "doc1", "filename": "test.md", "chunk_index": 0},
    ]

    original_key = settings.rerank_llm_api_key
    settings.rerank_llm_api_key = ""
    try:
        with (
            patch("src.tiny_rag.app.embedder.embed", return_value=[[0.1] * 768]),
            patch("src.tiny_rag.app.vector_store.search", return_value=mock_docs),
            patch("src.tiny_rag.app.bm25_retriever.search", return_value=[]),
            patch("src.tiny_rag.app.llm.rewrite", return_value="test query"),
            patch("src.tiny_rag.app.llm.generate_stream", return_value=iter(["answer"])),
            patch("src.tiny_rag.app.reranker.rerank") as mock_rerank,
        ):
            resp = client.post("/ask", json={"question": "test query"})
    finally:
        settings.rerank_llm_api_key = original_key

    assert resp.status_code == 200
    mock_rerank.assert_not_called()


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


def test_ask_sse_metadata(client):
    """SSE done 事件包含 original_question/rewritten/cached。"""
    from unittest.mock import patch

    with (
        patch("src.tiny_rag.app.embedder.embed", return_value=[[0.1] * 768]),
        patch("src.tiny_rag.app.vector_store.search", return_value=[
            {"text": "chunk", "doc_id": "doc1", "filename": "test.txt", "chunk_index": 0},
        ]),
        patch("src.tiny_rag.app.bm25_retriever.search", return_value=[]),
        patch("src.tiny_rag.app.llm.rewrite", return_value="test query"),
        patch("src.tiny_rag.app.llm.generate_stream", return_value=iter(["answer"])),
    ):
        resp = client.post("/ask", json={"question": "test question"})

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "event: done" in body
    # 提取 done 事件的 data 行
    lines = body.strip().split("\n")
    done_idx = lines.index("event: done")
    done_data = json.loads(lines[done_idx + 1].removeprefix("data: "))
    assert done_data.get("original_question") == "test question"
    assert done_data.get("rewritten") == "test query"
    assert done_data.get("cached") is False

