"""Chunker — split text into chunks with Markdown-aware semantic boundaries."""

import re
import logging
from dataclasses import dataclass

from .tokenizer import count_tokens, encode, decode

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    text: str
    heading_path: str = ""
    token_count: int = 0


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of at most chunk_size tokens with overlap.
    (Original token-sliding-window function, kept for backward compatibility.)
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


class MarkdownChunker:
    """Markdown-aware chunker that respects heading hierarchy,
    code blocks, and table boundaries."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[ChunkResult]:
        """Chunk markdown text by heading hierarchy.

        Args:
            text: Markdown text.

        Returns:
            List of ChunkResult with heading_path, text, token_count.
        """
        if not text.strip():
            return []

        lines = text.split("\n")
        protected = self._mark_protected(lines)
        sections = self._build_sections(lines, protected)
        return self._apply_size_management(sections)

    # ── Step 1: Mark protected regions ──────────────────────

    def _mark_protected(self, lines: list[str]) -> set[int]:
        """Return set of line indices protected from splitting."""
        protected: set[int] = set()
        i = 0
        while i < len(lines):
            # Code fence: allow up to 3 leading spaces
            m = re.match(r"^[ ]{0,3}(`{3,})\s*\S*$", lines[i])
            if m:
                fence = m.group(1)
                protected.add(i)
                i += 1
                while i < len(lines):
                    protected.add(i)
                    if lines[i].strip() == fence:
                        i += 1
                        break
                    i += 1
                continue

            # Table: consecutive lines starting with |
            if re.match(r"^\|.*\|", lines[i]):
                while i < len(lines) and re.match(r"^\|.*\|", lines[i]):
                    protected.add(i)
                    i += 1
                continue

            i += 1
        return protected

    # ── Step 2: Build sections by heading hierarchy ─────────

    def _build_sections(
        self, lines: list[str], protected: set[int]
    ) -> list[dict]:
        """Group lines into sections by heading hierarchy.

        Returns list of dicts: {heading_path, lines}
        """
        sections: list[dict] = []
        heading_stack: list[tuple[int, str]] = []
        current_lines: list[str] = []

        def flush():
            if current_lines:
                sections.append({
                    "heading_path": " > ".join(t[1] for t in heading_stack),
                    "lines": current_lines.copy(),
                })
                current_lines.clear()

        for i, line in enumerate(lines):
            if i in protected:
                current_lines.append(line)
                continue

            m = re.match(r"^(#{1,6})\s+(.+)$", line)
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))

            current_lines.append(line)

        if current_lines:
            sections.append({
                "heading_path": " > ".join(t[1] for t in heading_stack),
                "lines": current_lines.copy(),
            })

        return sections

    # ── Step 3: Size management ─────────────────────────────

    def _apply_size_management(self, sections: list[dict]) -> list[ChunkResult]:
        """Merge small sections, split large sections, apply overlap."""
        results: list[ChunkResult] = []

        def _chunk(text: str, hp: str) -> ChunkResult:
            return ChunkResult(
                text=text, heading_path=hp, token_count=count_tokens(text),
            )

        i = 0
        while i < len(sections):
            text = "\n".join(sections[i]["lines"])
            tokens = count_tokens(text)
            hp = sections[i]["heading_path"]

            # Small: merge with next, or previous if last
            if 0 < tokens < 100:
                if i + 1 < len(sections):
                    next_text = "\n".join(sections[i + 1]["lines"])
                    merged_hp = sections[i + 1]["heading_path"] if hp else hp
                    results.append(_chunk(text + "\n" + next_text, merged_hp))
                    i += 2
                    continue
                elif results:
                    prev = results.pop()
                    results.append(_chunk(prev.text + "\n" + text, hp))
                    i += 1
                    continue
                else:
                    results.append(_chunk(text, hp))

            # Within range: keep
            elif tokens <= self.chunk_size:
                results.append(_chunk(text, hp))

            # Too large: split
            else:
                results.extend(self._split_large(text, hp))

            i += 1

        return results

    # ── Large section splitting (paragraph → line → token) ──

    def _split_large(self, text: str, heading_path: str) -> list[ChunkResult]:
        """Split large text by paragraph, line, or token window."""
        paras = [p for p in text.split("\n\n") if p.strip()]
        if not paras:
            return self._split_by_tokens(text, heading_path)

        chunks: list[ChunkResult] = []
        current: list[str] = []
        current_tokens = 0

        for para in paras:
            pt = count_tokens(para)
            if current_tokens + pt <= self.chunk_size:
                current.append(para)
                current_tokens += pt
            else:
                if current:
                    chunks.append(ChunkResult(
                        text="\n\n".join(current),
                        heading_path=heading_path,
                        token_count=current_tokens,
                    ))
                current = [para]
                current_tokens = pt

        if current:
            chunks.append(ChunkResult(
                text="\n\n".join(current),
                heading_path=heading_path,
                token_count=current_tokens,
            ))

        # Single huge paragraph -> try line split
        if len(chunks) == 1 and current_tokens > self.chunk_size:
            return self._split_by_lines(text, heading_path)

        self._apply_overlap(chunks)
        return chunks

    def _split_by_lines(self, text: str, heading_path: str) -> list[ChunkResult]:
        """Split by newlines (second fallback)."""
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            return self._split_by_tokens(text, heading_path)

        chunks: list[ChunkResult] = []
        current: list[str] = []
        current_tokens = 0

        for line in lines:
            lt = count_tokens(line)
            if current_tokens + lt <= self.chunk_size:
                current.append(line)
                current_tokens += lt
            else:
                if current:
                    chunks.append(ChunkResult(
                        text="\n".join(current),
                        heading_path=heading_path,
                        token_count=current_tokens,
                    ))
                current = [line]
                current_tokens = lt

        if current:
            chunks.append(ChunkResult(
                text="\n".join(current),
                heading_path=heading_path,
                token_count=current_tokens,
            ))

        if len(chunks) == 1 and current_tokens > self.chunk_size:
            return self._split_by_tokens(text, heading_path)

        self._apply_overlap(chunks)
        return chunks

    def _split_by_tokens(self, text: str, heading_path: str) -> list[ChunkResult]:
        """Token sliding window (final fallback)."""
        tokens = encode(text)
        chunks: list[ChunkResult] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_str = decode(tokens[start:end])
            chunks.append(ChunkResult(
                text=chunk_str,
                heading_path=heading_path,
                token_count=end - start,
            ))
            if end >= len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ── Overlap ────────────────────────────────────────────

    def _apply_overlap(self, chunks: list[ChunkResult]) -> None:
        """Add overlap from previous chunk to current chunk."""
        if len(chunks) < 2 or self.chunk_overlap <= 0:
            return
        for i in range(1, len(chunks)):
            prev_tokens = encode(chunks[i - 1].text)
            if len(prev_tokens) <= self.chunk_overlap:
                overlap_text = chunks[i - 1].text
            else:
                overlap_text = decode(prev_tokens[-self.chunk_overlap:])
            chunks[i].text = overlap_text + chunks[i].text
            chunks[i].token_count = count_tokens(chunks[i].text)
