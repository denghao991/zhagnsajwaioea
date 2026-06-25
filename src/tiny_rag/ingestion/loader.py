"""Document loader — reads text files from disk or bytes."""

from pathlib import Path


def load_text(file_path: str | Path) -> str:
    """Read text content from a file path.

    Args:
        file_path: Path to a .txt file.

    Returns:
        File contents as a string.
    """
    return Path(file_path).read_text(encoding="utf-8")


def load_bytes(content: bytes) -> str:
    """Decode raw bytes to UTF-8 text.

    Args:
        content: Raw bytes from uploaded file.

    Returns:
        Decoded string.
    """
    return content.decode("utf-8")


