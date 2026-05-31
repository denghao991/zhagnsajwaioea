"""Hybrid search — merge vector + BM25 results via RRF."""

_K = 60  # RRF constant


def rrf_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    n_results: int = 5,
) -> list[dict]:
    """Merge two ranked result lists by Reciprocal Rank Fusion.

    Args:
        vector_results: Results from vector search (must have 'text' key).
        bm25_results: Results from BM25 search (must have 'text' key).
        n_results: Number of top results to return.

    Returns:
        Merged and sorted result list (text-deduplicated), preferring
        vector result dict when the same chunk appears in both lists.
    """
    # Build per-document RRF scores; prefer vector result on conflict
    scores: dict[str, float] = {}
    best_result: dict[str, dict] = {}

    for rank, result in enumerate(vector_results, start=1):
        text: str = result["text"]
        scores[text] = scores.get(text, 0.0) + 1.0 / (_K + rank)
        if text not in best_result:
            best_result[text] = result

    for rank, result in enumerate(bm25_results, start=1):
        text = result["text"]
        scores[text] = scores.get(text, 0.0) + 1.0 / (_K + rank)
        if text not in best_result:
            best_result[text] = result

    # Sort by RRF score descending
    ranked = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)[:n_results]

    return [best_result[t] for t in ranked]
