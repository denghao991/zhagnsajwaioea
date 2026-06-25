"""Tests for document loader module."""

from src.tiny_rag.ingestion.loader import load_text


def test_load_text_markdown(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\n\nHello **world**.", encoding="utf-8")
    content = load_text(str(md_file))
    assert "# Title" in content
    assert "**world**" in content
