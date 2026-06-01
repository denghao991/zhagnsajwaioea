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
