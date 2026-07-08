"""
语义缓存 —— 基于 embedding 相似度的答案缓存

核心认知：
- 不是精确字符串匹配 —— "OA 告警怎么配" 和 "优化顾问监控告警配置方法" 是同一个意思
- 把历史 query 的 embedding 存起来，新 query embedding 后算余弦相似度
- 超过阈值（如 0.92）视为语义等价，直接返回缓存答案
- 缓存命中时端到端延迟 < 100ms，跳过后面的检索和生成全部步骤

API 要点（面试用）：
- 缓存 key：query embedding (768-dim vector)
- 相似度：cosine similarity，阈值 0.92
- TTL + LRU 双重淘汰
- 与用户反馈联动：点踩不写入、点踩已缓存的立即删除、点赞加权
"""

import numpy as np
import time
from typing import Optional


class SemanticCache:
    """
    语义缓存实现（概念性代码）

    缓存字段：embedding, answer, hit_count, created_at, liked
    """

    def __init__(self, similarity_threshold: float = 0.92, ttl_seconds: int = 86400):
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        # 实际存储用 ChromaDB collection（query_cache），这里用 dict 示意
        self._store: dict[str, dict] = {}  # id → {embedding, answer, hit_count, ...}

    def cosine_similarity(self, a, b):
        """余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def lookup(self, query_embedding) -> Optional[str]:
        """
        查缓存：遍历所有缓存 entry，算余弦相似度，超阈值则命中
        2000 条缓存下遍历 < 1ms，不需要建索引
        """
        now = time.time()
        for entry_id, entry in self._store.items():
            # TTL 过期检查
            if now - entry["created_at"] > self.ttl:
                continue

            sim = self.cosine_similarity(query_embedding, entry["embedding"])
            if sim >= self.threshold:
                # 命中：hit_count +1，用于淘汰策略
                entry["hit_count"] += 1
                entry["last_accessed"] = now
                return entry["answer"]

        return None  # 未命中 → 走完整检索链路

    def write(self, query_embedding, answer):
        """写入缓存（详见 write_with_feedback 逻辑）"""
        import uuid
        entry_id = str(uuid.uuid4())
        self._store[entry_id] = {
            "embedding": query_embedding,
            "answer": answer,
            "hit_count": 1,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "liked": False,
        }

    def delete_by_query_text(self, query_text: str):
        """用户点踩了缓存命中的答案 → 立即删除该缓存条目（联动二）"""
        # 实际：通过 query embedding 定位并删除
        pass

    def apply_liked_boost(self, query_text: str, boost: int = 5):
        """用户点赞 → hit_count 额外 +N，更难被淘汰（联动三）"""
        pass


# --- 缓存写入的完整逻辑（联动一二三）---
def cache_write_with_feedback(query_embedding, answer, query_id, querylog):
    """
    延迟写 + 检查 user_click

    联动一（写入闸门）：
      LLM 生成完 → 返回答案给前端 → 等 5 分钟
      → 检查 QueryLog 中该 query_id 的 user_click
      → 如果不是"踩" → 写入缓存
      → 如果是"踩" → 不写入（避免错误答案固化）

    联动二（主动清除）：
      用户点踩了一个缓存命中的答案 → 立即删除该缓存条目
      → 多留一分钟就多一个用户被误导

    联动三（质量加权）：
      用户点赞 → hit_count +5（而不是 +1）
      → 点赞是比命中更强的正向信号
      → 容量淘汰时，被点赞的条目更难被淘汰
    """
    import time
    time.sleep(300)  # 等 5 分钟（实际是异步任务）

    click = querylog.get_user_click(query_id)
    if click == "踩":
        return  # 联动一：不写入

    cache.write(query_embedding, answer)
