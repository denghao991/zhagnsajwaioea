"""Tokenizer — tiktoken wrapper for text encoding/decoding."""

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return number of tokens in text."""
    return len(_ENCODING.encode(text, disallowed_special=()))


def encode(text: str) -> list[int]:
    """Encode text to token IDs."""
    return _ENCODING.encode(text, disallowed_special=())


def decode(tokens: list[int]) -> str:
    """Decode token IDs back to text."""
    return _ENCODING.decode(tokens)
