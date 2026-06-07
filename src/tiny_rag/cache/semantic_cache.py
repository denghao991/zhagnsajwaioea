"""Semantic cache — cache LLM responses by question embedding similarity."""

import json
import logging
import time

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """Cache LLM responses keyed by question embedding vectors.

    Uses ChromaDB collection with cosine distance for semantic matching.
    Threshold and max_entries are configurable; oldest entries are evicted
    when the limit is exceeded.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "cache",
        threshold: float = 0.03,
        max_entries: int = 500,
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._threshold = threshold
        self._max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def search(self, query_embedding: list[float]) -> dict | None:
        """Find cached entry by semantic similarity (cosine distance)."""
        if self._collection.count() == 0:
            self.misses += 1
            return None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["metadatas", "distances"],
        )

        if not results["ids"][0]:
            self.misses += 1
            return None

        distance = results["distances"][0][0]
        if distance > self._threshold:
            self.misses += 1
            return None

        self.hits += 1

        metadata = results["metadatas"][0][0]
        return {
            "question": metadata.get("question", ""),
            "answer": metadata.get("answer", ""),
            "sources": json.loads(metadata.get("sources", "[]")),
        }

    def put(
        self,
        question: str,
        answer: str,
        embedding: list[float],
        sources: list[dict],
        entry_id: str,
    ) -> None:
        """Store a cache entry, evicting oldest if over max_entries."""
        self._collection.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            metadatas=[{
                "question": question,
                "answer": answer,
                "sources": json.dumps(sources),
                "created_at": time.time(),
            }],
        )
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries when cache exceeds max_entries."""
        count = self._collection.count()
        if count <= self._max_entries:
            return

        all_data = self._collection.get(include=["metadatas"])
        id_time = []
        for sid, meta in zip(all_data["ids"], all_data["metadatas"]):
            id_time.append((sid, meta.get("created_at", 0)))
        id_time.sort(key=lambda x: x[1])

        excess = count - self._max_entries
        delete_ids = [sid for sid, _ in id_time[:excess]]
        self._collection.delete(ids=delete_ids)

    def clear(self) -> None:
        """Delete all cache entries."""
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

    def get_stats(self) -> dict:
        """Return cache statistics."""
        total = self.hits + self.misses
        return {
            "cache_entries": self._collection.count(),
            "total_requests": total,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total * 100, 1) if total else 0.0,
            "threshold": self._threshold,
            "max_entries": self._max_entries,
        }
