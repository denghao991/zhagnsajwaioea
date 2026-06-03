"""Tests for LLM client."""

from unittest.mock import patch, MagicMock
from collections.abc import Generator

from src.tiny_rag.generation.llm import LLMClient


def _mock_chunk(content: str | None):
    """Helper to create a mock streaming chunk."""
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    chunk.choices = [MagicMock(delta=delta)]
    return chunk


def test_generate_returns_answer():
    client = LLMClient(base_url="http://fake", api_key="test", model="test-model")

    mock_message = MagicMock()
    mock_message.content = "Based on the documents, the answer is 42."

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(client._client.chat.completions, "create", return_value=mock_response):
        result = client.generate(question="What is the answer?", context="The answer is 42.")

    assert result == "Based on the documents, the answer is 42."


def test_generate_stream_yields_tokens():
    client = LLMClient(base_url="http://fake", api_key="test", model="test-model")

    chunks = [
        _mock_chunk("Hello"),
        _mock_chunk(" "),
        _mock_chunk("world"),
        _mock_chunk(None),  # signal end
    ]

    with patch.object(client._client.chat.completions, "create", return_value=chunks):
        tokens = list(client.generate_stream(question="test", context="test context"))

    assert tokens == ["Hello", " ", "world"]


def test_generate_stream_returns_generator():
    client = LLMClient(base_url="http://fake", api_key="test", model="test-model")

    with patch.object(client._client.chat.completions, "create", return_value=[]):
        result = client.generate_stream(question="test", context="test")

    assert isinstance(result, Generator)


def test_rewrite_expands_abbreviation():
    client = LLMClient(base_url="http://fake", api_key="test", model="test-model")

    mock_message = MagicMock()
    mock_message.content = "优化顾问(OA)有哪些功能"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(client._client.chat.completions, "create", return_value=mock_response):
        result = client.rewrite("OA有哪些功能")

    assert result == "优化顾问(OA)有哪些功能"


def test_rewrite_fallback_on_error():
    client = LLMClient(base_url="http://fake", api_key="test", model="test-model")

    with patch.object(client._client.chat.completions, "create", side_effect=Exception("API error")):
        result = client.rewrite("测试问题")

    assert result == "测试问题"


def test_rewrite_fallback_on_empty():
    client = LLMClient(base_url="http://fake", api_key="test", model="test-model")

    mock_message = MagicMock()
    mock_message.content = None
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(client._client.chat.completions, "create", return_value=mock_response):
        result = client.rewrite("测试问题")

    assert result == "测试问题"
