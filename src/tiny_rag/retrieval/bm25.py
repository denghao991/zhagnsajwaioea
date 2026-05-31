"""BM25 keyword retriever — exact-match complement to vector search."""

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 index over document chunks for keyword-style retrieval.

    Maintains an independent copy of chunk texts alongside VectorStore.
    Rebuilds the BM25 index from scratch on each add.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def add_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[str],
    ) -> None:
        """Add document chunks to the BM25 index.

        Args:
            doc_id: Unique document identifier.
            filename: Original filename.
            chunks: List of text chunks.
        """
        self._chunks.extend(chunks)
        self._metadatas.extend(
            {"doc_id": doc_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        )

        tokenized_corpus = [self._tokenize(c) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Search by BM25 score.

        Args:
            query: Raw query string.
            n_results: Number of top results.

        Returns:
            List of result dicts with keys: doc_id, filename, chunk_index, text, score.
        """
        if not self._bm25 or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Top N by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:n_results]

        return [
            {
                "doc_id": self._metadatas[i]["doc_id"],
                "filename": self._metadatas[i]["filename"],
                "chunk_index": self._metadatas[i]["chunk_index"],
                "text": self._chunks[i],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]

    def clear(self) -> None:
        """Reset the index (e.g. after document deletion)."""
        self._chunks.clear()
        self._metadatas.clear()
        self._bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace+punctuation tokenizer for Chinese text."""
        return text.lower().split()
