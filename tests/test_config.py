"""Tests for config.py – TERM_MAP and REWRITE_PATTERN."""

from src.tiny_rag.config import TERM_MAP, REWRITE_PATTERN


class TestTermMap:
    """TERM_MAP provides abbreviation-to-full-name mappings."""

    def test_term_map_contains_expected_keys(self):
        assert "OA" in TERM_MAP
        assert "CSS" in TERM_MAP
        assert "CCE" in TERM_MAP

    def test_term_map_values_are_non_empty_strings(self):
        for key, value in TERM_MAP.items():
            assert isinstance(key, str) and key, f"Key {key!r} is invalid"
            assert isinstance(value, str) and value, f"Value for {key!r} is empty"

    def test_term_map_values_contain_abbreviation(self):
        """Each mapped value contains the abbreviation so round-tripping works."""
        assert "OA" in TERM_MAP["OA"]
        assert "CSS" in TERM_MAP["CSS"]
        assert "CCE" in TERM_MAP["CCE"]


class TestRewritePattern:
    """REWRITE_PATTERN is a non-empty docstring-style constant."""

    def test_rewrite_pattern_is_string(self):
        assert isinstance(REWRITE_PATTERN, str)

    def test_rewrite_pattern_is_not_empty(self):
        assert len(REWRITE_PATTERN) > 0

    def test_rewrite_pattern_contains_example(self):
        assert "CSS" in REWRITE_PATTERN or "云服务" in REWRITE_PATTERN


class TestCacheConfig:
    def test_cache_threshold_default(self):
        from src.tiny_rag.config import CACHE_THRESHOLD
        assert isinstance(CACHE_THRESHOLD, float)
        assert 0 < CACHE_THRESHOLD < 1

    def test_cache_max_entries_default(self):
        from src.tiny_rag.config import CACHE_MAX_ENTRIES
        assert isinstance(CACHE_MAX_ENTRIES, int)
        assert CACHE_MAX_ENTRIES > 0
