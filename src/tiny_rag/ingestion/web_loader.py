"""Web page loader — fetch HTML pages and convert to Markdown."""

import logging
import re
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import html2text
import httpx

logger = logging.getLogger(__name__)

_IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")


@dataclass
class PageResult:
    url: str
    markdown: str
    depth: int


class WebLoader:
    """BFS web crawler that converts HTML pages to Markdown.

    Args:
        max_depth: Maximum link-following depth (default 20).
        request_timeout: HTTP request timeout in seconds (default 30).
    """

    def __init__(self, max_depth: int = 20, request_timeout: float = 30.0):
        self.max_depth = max_depth
        self.request_timeout = request_timeout
        self._converter = html2text.HTML2Text()
        self._converter.body_width = 0
        self._converter.skip_internal_links = False
        self._converter.protect_links = True

    def load(self, start_url: str, max_depth: int | None = None) -> list[PageResult]:
        """BFS crawl from *start_url*, return all fetched pages.

        Args:
            start_url: The URL to start crawling from.
            max_depth: Override for instance max_depth (optional).

        Returns:
            List of PageResult (url, markdown, depth).
        """
        effective_depth = max_depth if max_depth is not None else self.max_depth
        results: list[PageResult] = []
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()

        norm = self._normalize_url(start_url, start_url)
        if not norm:
            logger.warning("Invalid start URL: %s", start_url)
            return results

        queue.append((norm, 0))
        visited.add(norm)

        with httpx.Client(timeout=self.request_timeout, follow_redirects=True) as client:
            while queue:
                url, depth = queue.popleft()

                try:
                    resp = client.get(url)
                    resp.raise_for_status()

                    content_type = resp.headers.get("content-type", "").lower()
                    if "text/html" not in content_type:
                        logger.info("Skipping non-HTML: %s (%s)", url, content_type)
                        continue

                    html = resp.text
                    markdown = self._converter.handle(html)
                    markdown = _IMAGE_PATTERN.sub("", markdown).strip()

                    results.append(PageResult(url=url, markdown=markdown, depth=depth))

                    # Extract links for next BFS level
                    if depth < effective_depth:
                        for href in self._extract_links(html):
                            absolute = self._normalize_url(href, url)
                            if absolute and absolute not in visited:
                                visited.add(absolute)
                                queue.append((absolute, depth + 1))

                except Exception as e:
                    logger.warning("Failed to fetch %s: %s", url, e)

        return results

    def _extract_links(self, html: str) -> list[str]:
        """Extract all href values from <a> tags in HTML."""
        links: list[str] = []
        for m in re.finditer(r'<a\s+[^>]*href="([^"]*)"', html, re.IGNORECASE):
            links.append(m.group(1))
        for m in re.finditer(r"<a\s+[^>]*href='([^']*)'", html, re.IGNORECASE):
            links.append(m.group(1))
        return links

    @staticmethod
    def _normalize_url(href: str, base: str) -> str | None:
        """Resolve relative URL and return normalized absolute URL, or None."""
        absolute = urljoin(base, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            return None
        qs = f"?{parsed.query}" if parsed.query else ""
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}{qs}"
