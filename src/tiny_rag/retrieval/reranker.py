"""Rerank client — Cross-encoder re-ranking via DashScope Rerank API."""

import json
import logging
from typing import Any

import httpx


logger = logging.getLogger(__name__)


class RerankClient:
    """Re-rank retrieved documents using DashScope Rerank API.

    Wraps the ``POST /api/v1/services/rerank/text-rerank/text-rerank``
    endpoint.  Falls back to the original order on any API error.

    Note: Unlike EmbeddingClient (which uses an OpenAI-compatible endpoint),
    the DashScope Rerank API has its own non-OpenAI-compatible format, so we
    call it directly via HTTP POST instead of through the OpenAI SDK.
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Re-rank *documents* by relevance to *query*.

        Args:
            query: User question.
            documents: Result dicts from RRF merge (must have ``"text"`` key).
            top_n: Number of top results to return.

        Returns:
            Re-ranked result list (same dict format as input), with
            ``"score"`` updated from the API.
        """
        if not documents or not query.strip():
            return documents[:top_n] if top_n is not None else documents

        try:
            texts = [d["text"] for d in documents]
        except KeyError:
            logger.warning("Rerank input missing 'text' key, falling back to original order")
            return documents[:top_n] if top_n is not None else documents

        try:
            payload: dict[str, Any] = {
                "model": self._model,
                "input": {"query": query, "documents": texts},
                "parameters": {"top_n": min(top_n, len(texts))},
            }
            resp = httpx.post(
                f"{self._base_url}/api/v1/services/rerank/text-rerank/text-rerank",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data["output"]["results"]
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            logger.warning("Rerank API call failed, falling back to original order", exc_info=True)
            return documents[:top_n] if top_n is not None else documents

        ranked: list[dict[str, Any]] = []
        for item in results:
            idx = item["index"]
            doc = dict(documents[idx])
            doc["score"] = item["relevance_score"]
            ranked.append(doc)

        ranked.sort(key=lambda d: d["score"], reverse=True)
        return ranked
