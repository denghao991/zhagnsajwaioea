"""Tests for embedding client."""

from unittest.mock import patch, MagicMock
from src.tiny_rag.embedding.client import EmbeddingClient


def test_embed_returns_list_of_vectors():
    client = EmbeddingClient(base_url="http://fake", api_key="test", model="test-model")

    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]

    with patch.object(client._client.embeddings, "create", return_value=mock_response):
        result = client.embed(["hello", "world"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]
