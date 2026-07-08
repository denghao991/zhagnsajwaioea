"""
BM25 稀疏检索 —— jieba 分词 + rank_bm25 评分 + ChromaDB sparse 存储

核心认知：
- BM25 是纯统计算法，不需要 GPU、不需要模型推理
- ChromaDB 的 sparse index 存稀疏向量，负责 RRF 融合
- BM25 的评分计算（TF 饱和 + IDF 加权 + 长度归一化）在外部用 rank_bm25
- 不是 ChromaDB 内置 BM25 —— ChromaDB 内置的用 snowballstemmer，不支持 jieba

API 要点（面试用）：
- jieba.cut(text) → 分词
- jieba.load_userdict(path) → 加载自定义词库
- rank_bm25.BM25Okapi(corpus_tokens) → 构建 BM25 模型
- bm25.get_scores(query_tokens) → 算分
- ChromaDB sparse vectors: collection.add(embeddings=sparse_vectors)
"""

import jieba
import numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi
import chromadb


# --- 1. jieba 分词 + 自定义词库 ---
jieba.load_userdict("data/jieba_dict.txt")  # 内部专有名词：多AZ、OA、优化顾问...

# 运行时热加载（改词库后调一次即可，不用重启）
def reload_jieba_dict():
    """改词库文件后，调此函数热加载，~30s 内全节点生效"""
    jieba.load_userdict("data/jieba_dict.txt")


# --- 2. 离线阶段：构建 BM25 倒排索引 + 存入 ChromaDB ---
def build_bm25_index(chunks: list[dict]):
    """
    chunks: [{"id": ..., "content": ..., "metadata": ...}, ...]
    返回: BM25Okapi 模型 (内存倒排索引)
    """
    # Step 1: jieba 分词 —— 走的是我们维护的自定义词库
    tokenized = [list(jieba.cut(chunk["content"])) for chunk in chunks]
    # 结果示例：['OA', '监控告警', '配置', '步骤', '：', '1', '.', '登录', '控制台', ...]

    # Step 2: rank_bm25 构建倒排索引
    #   - TF 饱和：term_freq / (term_freq + k1*(1 - b + b*doc_len/avg_len))
    #   - IDF 加权：log((N - df + 0.5) / (df + 0.5) + 1)
    #   - 长度归一化：短文档中词出现一次比长文档中更"值钱"
    #   k1 默认 1.5, b 默认 0.75 —— 经典参数，整个学界都在用
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)

    # Step 3: 为每个 chunk 生成稀疏向量（词 → BM25 权重），存入 ChromaDB
    # 这部分是"把 BM25 结果用 ChromaDB 存起来，方便后续 RRF 融合"
    # 实际生产中是每次启动自动重建，毫秒级
    return bm25


# --- 3. 在线检索 ---
def bm25_search(query: str, bm25: BM25Okapi, chunks: list[dict], top_k: int = 5):
    """
    在线 BM25 检索 —— 全内存操作，~2ms
    返回: [{"id": ..., "content": ..., "metadata": ..., "score": ...}, ...]
    """
    # jieba 分词
    query_tokens = list(jieba.cut(query))

    # BM25 打分
    scores = bm25.get_scores(query_tokens)  # ndarray, shape=(N_chunks,)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                **chunks[idx],
                "score": float(scores[idx])
            })
    return results


# --- BM25 分数是怎么算的（概念性代码）---
def bm25_score_explained(term_freq: int, doc_len: int, avg_doc_len: float,
                          num_docs: int, doc_freq: int, k1=1.5, b=0.75):
    """
    单个 term 对单个 doc 的 BM25 分数（简化说明版）：
    - 第一项 IDF：log((N - df + 0.5) / (df + 0.5) + 1)
      稀有词（df 小）→ IDF 大 → 分数高。如"多AZ"只出现在3篇文档，IDF 很高
      常见词（df 大）→ IDF 小 → 分数低。如"配置"出现在200篇，几乎无区分度
    - 第二项 TF 饱和：tf*(k1+1) / (tf + k1*(1 - b + b*doc_len/avg_len))
      词频越高分越高，但增速递减——出现10次不是出现1次的10倍分数
      k1 控制饱和速度，b 控制长度归一化强度
    """
    # IDF
    idf = np.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    # TF 饱和 + 文档长度归一化
    tf_norm = (term_freq * (k1 + 1)) / (
        term_freq + k1 * (1 - b + b * doc_len / avg_doc_len)
    )
    return idf * tf_norm
