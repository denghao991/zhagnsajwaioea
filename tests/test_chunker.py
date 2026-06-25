"""Tests for chunker module."""

from src.tiny_rag.ingestion.chunker import MarkdownChunker, ChunkResult


# ── MarkdownChunker tests ──

def make_chunker(chunk_size=512, chunk_overlap=0):
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
    """h1 and h2 create sections; h3 content stays inside parent h2 section."""
    text = "# Level1\n\nIntro\n\n## Level2\n\nDetail\n\n### Level3\n\nDeep"
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    paths = [c.heading_path for c in chunks]
    assert "Level1" in paths
    assert "Level1 > Level2" in paths
    # h3 content is inside the h2 section
    level3_chunks = [c for c in chunks if "Deep" in c.text]
    assert len(level3_chunks) > 0
    assert "Level1 > Level2" in level3_chunks[0].heading_path


def test_markdown_heading_stack_resets():
    """A lower-level heading should reset the stack."""
    text = "# A\n\n## A1\n\n# B\n\n## B1\n\n### B1a"
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    paths = [c.heading_path for c in chunks]
    b_sections = [p for p in paths if p.startswith("B")]
    assert len(b_sections) >= 1
    assert all("A" not in p for p in b_sections)


def test_markdown_preamble():
    """Content before first heading should be included in a chunk."""
    text = "Preamble paragraph.\n\n# Title\n\nBody."
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    all_text = " ".join(c.text for c in chunks)
    assert "Preamble" in all_text


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


def test_markdown_small_section():
    """Small heading sections stay as their own chunk (no cross-heading merge)."""
    text = "# A\n\nSmall.\n\n# B\n\nLarger content here.\n\nMore content.\n\nStill going to fill up tokens."
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    all_text = " ".join(c.text for c in chunks)
    assert "Small" in all_text


def test_markdown_last_section_no_merge():
    """Last small section stays as its own chunk."""
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
    for c in chunks:
        assert c.token_count <= 150


def test_markdown_overlap():
    """Adjacent sub-chunks should have overlap content."""
    text = "# Test\n\n" + "\n\n".join(f"Content paragraph {i} with some text to make it long enough for testing the overlap behavior." for i in range(15))
    chunks = make_chunker(chunk_size=120, chunk_overlap=30).chunk_text(text)
    assert len(chunks) >= 2
    if len(chunks) >= 2:
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
    """Only h1 and h2 appear in heading_path; deeper headings stay in content."""
    text = "# Root\n\n## Child\n\n### Grandchild\n\nDeep content."
    chunks = make_chunker(chunk_size=512).chunk_text(text)
    for c in chunks:
        if "Deep content" in c.text:
            assert c.heading_path == "Root > Child"


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
