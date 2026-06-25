"""Chunker — split text into chunks with Markdown-aware semantic boundaries."""

from dataclasses import dataclass

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .tokenizer import count_tokens


@dataclass
class ChunkResult:
    text: str
    heading_path: str = ""
    token_count: int = 0


class MarkdownChunker:
    """Markdown-aware chunker using LangChain splitters.

    Splits by heading (#, ##) then uses RecursiveCharacterTextSplitter
    for chunk_size enforcement.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0, strip_headers: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
            ],
            strip_headers=strip_headers,
        )
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=count_tokens,
        )

    def chunk_text(self, text: str) -> list[ChunkResult]:
        if not text.strip():
            return []

        docs = self._md_splitter.split_text(text)
        results: list[ChunkResult] = []

        for doc in docs:
            hp = " > ".join(v for v in doc.metadata.values())
            sub_docs = self._recursive_splitter.split_documents([doc])

            for sd in sub_docs:
                content = sd.page_content
                if hp:
                    content = f"{hp} > {content}"
                results.append(ChunkResult(
                    text=content,
                    heading_path=hp,
                    token_count=count_tokens(content),
                ))

        return results
