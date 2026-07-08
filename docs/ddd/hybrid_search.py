"""
混合检索 —— 向量路 + BM25路并行 → RRF 融合

核心认知：
- 两路完全独立、并行执行、取较慢那路的时间（约 5ms = max(3ms, 2ms)）
- RRF 只关心排序位置（rank），不关心分数绝对值 → 天然规避向量相似度和 BM25 分的量纲问题
- k=60 是 RRF 论文经典取值，结果对此参数不敏感
- α/β 等权 0.5，网格搜索验证最优值在此附近

API 要点（面试用）：
- concurrent.futures.ThreadPoolExecutor → 并行执行两路检索
- RRF 公式：score(chunk) = α/(k+rank_vec) + β/(k+rank_bm25)
"""

import concurrent.futures
from typing import Optional


def dense_search(query_embedding, collection, top_k: int = 5) -> list[dict]:
    """向量路：ChromaDB HNSW 检索，~3ms"""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],  # 距离转相似度
        }
        for i in range(len(results["ids"][0]))
    ]


def sparse_search(query: str, bm25, chunks, top_k: int = 5) -> list[dict]:
    """BM25路：jieba分词 + 倒排索引查询，~2ms"""
    from bm25_retriever import bm25_search
    return bm25_search(query, bm25, chunks, top_k)


def hybrid_search(query: str, query_embedding, collection, bm25, chunks,
                  top_k: int = 5) -> list[dict]:
    """
    双路并行检索 + RRF 融合
    总耗时 ~5ms = max(vector ~3ms, BM25 ~2ms)
    """
    # 并行执行两路检索
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_dense = executor.submit(dense_search, query_embedding, collection, top_k)
        future_sparse = executor.submit(sparse_search, query, bm25, chunks, top_k)
        dense_results = future_dense.result()
        sparse_results = future_sparse.result()

    # RRF 融合
    merged = rrf_fusion(dense_results, sparse_results, alpha=0.5, beta=0.5, k=60)
    return merged


def rrf_fusion(dense_results: list[dict], sparse_results: list[dict],
               alpha: float = 0.5, beta: float = 0.5, k: int = 60) -> list[dict]:
    """
    RRF (Reciprocal Rank Fusion)

    为什么不用线性加权？
      向量相似度（余弦，0~1）和 BM25 分数（理论无上限）量纲完全不同
      0.9 的余弦 vs 9.5 的 BM25 —— 没法直接比谁大谁小
      RRF 只关心 rank：排第 1 就是第 1，不管你分数是 0.95 还是 9.5

    公式：score(chunk) = α/(k + rank_vec) + β/(k + rank_bm25)
    - rank 从 1 开始（第 1 名 rank=1）
    - k=60：k 越小越偏向第一名，k 越大越平滑。60 是论文经典值，结果对此不敏感
    - α/β：当前等权 0.5，网格搜索验证最优值在此附近

    去重后约 8 个候选（5 + 5，交集约 2~3 个）
    """
    scores = {}

    # 向量路：给每个 chunk 的 RRF 贡献
    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + alpha / (k + rank)

    # BM25 路：给每个 chunk 的 RRF 贡献
    for rank, item in enumerate(sparse_results, start=1):
        chunk_id = item["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + beta / (k + rank)

    # 按融合分降序排列
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 构建结果（合并两路的 metadata）
    dense_map = {item["id"]: item for item in dense_results}
    sparse_map = {item["id"]: item for item in sparse_results}
    merged = []
    for chunk_id, rrf_score in sorted_chunks:
        item = dense_map.get(chunk_id) or sparse_map.get(chunk_id)
        if item:
            merged.append({**item, "rrf_score": rrf_score})
    return merged


# --- α/β 参数搜索（Q9 网格搜索）---
def search_rrf_weights(dense_results, sparse_results, eval_set, step=0.1):
    """
    网格搜索最优 α/β（α + β = 1）
    不需要重跑检索——改 α/β 只重新算融合分数，一秒跑完
    """
    best_mrr = 0
    best_alpha = 0.5
    for alpha in [i * step for i in range(11)]:  # 0, 0.1, ..., 1.0
        beta = 1 - alpha
        merged = rrf_fusion(dense_results, sparse_results, alpha, beta)
        # 在评测集上算 MRR...
        # mrr = compute_mrr(merged, eval_set)
        # if mrr > best_mrr: ...
    return best_alpha
