"""Vector store — ChromaDB wrapper for document storage and retrieval."""

from collections.abc import Sequence

import chromadb
from chromadb.config import Settings

from src.tiny_rag.ingestion.chunker import ChunkResult


class VectorStore:
    """ChromaDB-based vector store for RAG document storage."""

    def __init__(self, persist_dir: str) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="documents",
        )

    def add_document(
        self,
        doc_id: str,
        filename: str,
        chunks: list[ChunkResult],
        embeddings: list[list[float]],
    ) -> None:
        """Store document chunks with their embeddings.

        Args:
            doc_id: Unique document identifier.
            filename: Original filename.
            chunks: List of ChunkResult objects.
            embeddings: List of embedding vectors, one per chunk.
        """
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "heading_path": chunks[i].heading_path,
            }
            for i in range(len(chunks))
        ]

        self._collection.add(
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict]:
        """Search for most similar chunks by embedding.

        Args:
            query_embedding: Embedding vector of the query.
            n_results: Number of top results to return.

        Returns:
            List of result dicts with keys: doc_id, filename, chunk_index,
            heading_path, text, distance.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            output.append({
                "doc_id": meta.get("doc_id", ""),
                "filename": meta.get("filename", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "heading_path": meta.get("heading_path", ""),
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def list_documents(self) -> list[dict]:
        """List all unique documents and their chunk counts.

        Returns:
            List of dicts with keys: id, chunks_count.
        """
        all_meta = self._collection.get(include=["metadatas"])
        doc_map: dict[str, int] = {}
        for meta in all_meta["metadatas"]:
            doc_id = meta["doc_id"]
            doc_map[doc_id] = doc_map.get(doc_id, 0) + 1

        return [
            {"id": doc_id, "chunks": count}
            for doc_id, count in sorted(doc_map.items())
        ]
