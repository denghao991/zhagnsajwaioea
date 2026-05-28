"""Tests for document loader module."""

from src.tiny_rag.ingestion.loader import load_text, load_pdf


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
    assert "Hello PDF" in text


def test_load_text_markdown(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\n\nHello **world**.", encoding="utf-8")
    content = load_text(str(md_file))
    assert "# Title" in content
    assert "**world**" in content
