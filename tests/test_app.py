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
