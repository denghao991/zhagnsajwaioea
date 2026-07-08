"""
Rerank 重排序 —— Qwen3-rerank cross-encoder

核心认知：
- cross-encoder vs bi-encoder：
  - bi-encoder（如 text-embedding-v3）：query 和 doc 分别编码，速度快但交互弱
  - cross-encoder（如 Qwen3-rerank）：query 和 doc 拼接后做全注意力推理，精度高但慢
- 流程：RRF 去重后 ~8 个候选 → 每个逐个过 cross-encoder → 按分数降序 → 取 top-5
- 延迟：~100ms（8 个 × 每对 ~12-15ms cross-encoder 推理）
- 消融实验：加 Rerank 后 MRR 0.831 → 0.868

API 要点（面试用）：
- 内部 gRPC API：query + document 拼接 → 全注意力推理 → 相关性分数
- 输入 8 个候选，输出 top-5
"""

from typing import Optional


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    用 cross-encoder 对候选 chunk 做精排

    Args:
        query: 用户原始问题
        candidates: RRF 融合后的候选列表（~8 个）
        top_k: 最终保留数量

    Returns:
        按相关性分数降序排列的 top-k chunk 列表
    """
    # 实际实现：逐个候选将 (query, chunk_content) 拼接后调 Qwen3-rerank API
    scored = []
    for c in candidates:
        # 将 query 和 chunk 拼接，cross-encoder 做全注意力推理
        # input_text = f"Query: {query}\nDocument: {c['content']}"
        # score = rerank_api.score(query, c["content"])  # gRPC 调用
        score = _call_rerank_api(query, c["content"])
        scored.append({**c, "rerank_score": score})

    # 按分数降序
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


def _call_rerank_api(query: str, document: str) -> float:
    """
    内部 Qwen3-rerank API 调用（概念性代码）

    Cross-encoder 和 bi-encoder 的本质区别：
    ┌─────────────────┐    ┌──────────────────────┐
    │  Bi-encoder      │    │  Cross-encoder       │
    │  query ─→ vec_q  │    │  [query, doc] ─→ 分  │
    │  doc  ─→ vec_d   │    │  ↑ query 的每个 token │
    │  cos(vec_q,vec_d)│    │  和 doc 的每个 token  │
    │  独立编码，快     │    │  做了全注意力交互     │
    │  ~1ms/对          │    │  ~12-15ms/对          │
    └─────────────────┘    └──────────────────────┘
    """
    # 实际走内部 gRPC API，这里只展示调用形态
    # response = rerank_client.rerank(
    #     model="qwen3-rerank",
    #     query=query,
    #     documents=[document],
    # )
    # return response.scores[0]
    pass


# --- 消融实验数据（Q11）---
def rerank_ablation_summary():
    """Rerank 消融实验对比"""
    return {
        "无 Rerank (RRF 直接 top-5)": {"MRR": 0.831, "Recall@5": 0.892},
        "有 Rerank (Qwen3-rerank)":   {"MRR": 0.868, "Recall@5": 0.914},
        "代价": "增加约 100ms 延迟，日均 100 问完全可接受",
    }
