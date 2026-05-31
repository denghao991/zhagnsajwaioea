"""Tests for RerankClient."""

from unittest.mock import Mock, patch

import httpx
import pytest

from src.tiny_rag.retrieval.reranker import RerankClient


@pytest.fixture
def client() -> RerankClient:
    return RerankClient(
        base_url="https://dashscope.aliyuncs.com",
        api_key="sk-test",
        model="gte-rerank",
    )


def test_rerank_success(client: RerankClient) -> None:
    """Normal API call returns re-ranked results."""
    docs = [
        {"text": "CSS是云搜索服务", "doc_id": "d1"},
        {"text": "OBS是对象存储", "doc_id": "d2"},
        {"text": "ECS是弹性计算", "doc_id": "d3"},
    ]
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "output": {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.80},
                {"index": 1, "relevance_score": 0.30},
            ]
        },
        "usage": {"total_tokens": 30},
    }
    mock_resp.raise_for_status = Mock(return_value=None)

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        result = client.rerank("ECS是什么", docs, top_n=3)

    # 验证 API 调用参数
    call_args = mock_post.call_args
    assert call_args[0][0].endswith("/api/v1/services/rerank/text-rerank/text-rerank")
    sent = call_args[1]["json"]
    assert sent["input"]["query"] == "ECS是什么"
    assert sent["input"]["documents"] == ["CSS是云搜索服务", "OBS是对象存储", "ECS是弹性计算"]
    assert sent["parameters"]["top_n"] == 3

    # 验证排序: ECS → CSS → OBS
    assert len(result) == 3
    assert result[0]["doc_id"] == "d3"  # ECS, score 0.95
    assert result[0]["score"] == 0.95
    assert result[1]["doc_id"] == "d1"  # CSS, score 0.80
    assert result[1]["score"] == 0.80


def test_rerank_empty_documents(client: RerankClient) -> None:
    """Empty document list returns empty."""
    assert client.rerank("test", []) == []


def test_rerank_empty_query(client: RerankClient) -> None:
    """Empty query returns original docs unchanged."""
    docs = [{"text": "some text", "doc_id": "d1"}]
    result = client.rerank("", docs, top_n=5)
    assert result == docs


def test_rerank_api_error_fallback(client: RerankClient) -> None:
    """API error falls back to original order."""
    docs = [
        {"text": "doc a", "doc_id": "d1"},
        {"text": "doc b", "doc_id": "d2"},
        {"text": "doc c", "doc_id": "d3"},
    ]
    with patch("httpx.post", side_effect=httpx.ConnectError("connection failed")):
        result = client.rerank("test query", docs, top_n=2)

    # 回退到原始顺序的前 top_n 条
    assert len(result) == 2
    assert result[0]["doc_id"] == "d1"
    assert result[1]["doc_id"] == "d2"


def test_rerank_top_n_less_than_total(client: RerankClient) -> None:
    """top_n returns only N results, sorted by score."""
    docs = [{"text": f"doc {i}", "doc_id": f"d{i}"} for i in range(5)]
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "output": {
            "results": [
                {"index": 3, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
            ]
        },
        "usage": {"total_tokens": 20},
    }
    mock_resp.raise_for_status = Mock(return_value=None)

    with patch("httpx.post", return_value=mock_resp):
        result = client.rerank("query", docs, top_n=2)

    assert len(result) == 2
    assert result[0]["doc_id"] == "d3"  # index 3, score 0.9
    assert result[1]["doc_id"] == "d1"  # index 1, score 0.7
    assert result[0]["score"] == 0.9
    assert result[1]["score"] == 0.7


def test_rerank_returns_ordered_by_score(client: RerankClient) -> None:
    """Results are sorted by relevance_score descending."""
    docs = [
        {"text": "doc a", "doc_id": "d1"},
        {"text": "doc b", "doc_id": "d2"},
        {"text": "doc c", "doc_id": "d3"},
    ]
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "output": {
            "results": [
                {"index": 1, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.7},
            ]
        },
        "usage": {"total_tokens": 30},
    }
    mock_resp.raise_for_status = Mock(return_value=None)

    with patch("httpx.post", return_value=mock_resp):
        result = client.rerank("query", docs, top_n=3)

    # scores descending: 0.9, 0.7, 0.5
    assert [r["doc_id"] for r in result] == ["d1", "d3", "d2"]
    assert [r["score"] for r in result] == [0.9, 0.7, 0.5]
