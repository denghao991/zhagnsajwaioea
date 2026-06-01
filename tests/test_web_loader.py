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

    def normalize(href, base):
        return href if href.startswith("http") else None

    loader._normalize_url = normalize

    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.side_effect = side_effect

        results = loader.load("https://example.com")

    assert len(results) >= 2


def test_load_depth_limit():
    loader = WebLoader(max_depth=0)
    loader._normalize_url = lambda h, b: h if h.startswith("http") else None

    with patch("httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.return_value = _mock_response(SAMPLE_HTML)

        results = loader.load("https://example.com")

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
