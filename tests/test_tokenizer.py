"""Tests for tokenizer module."""

from src.tiny_rag.ingestion.tokenizer import count_tokens, encode, decode


def test_count_tokens_returns_positive_int():
    result = count_tokens("Hello, world!")
    assert isinstance(result, int)
    assert result > 0


def test_encode_decode_roundtrip():
    text = "The quick brown fox jumps over the lazy dog."
    tokens = encode(text)
    decoded = decode(tokens)
    assert decoded == text


def test_count_tokens_empty_string():
    assert count_tokens("") == 0


def test_encode_empty_string():
    assert encode("") == []


def test_decode_empty_list():
    assert decode([]) == ""
