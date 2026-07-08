"""
ChromaDB 向量存储 —— PersistentClient 嵌入模式

API 要点（面试用）：
- chromadb.PersistentClient(path) → 进程内调用，数据文件存本地磁盘
- client.get_or_create_collection(name, metadata, embedding_function)
- collection.add(ids, documents, metadatas, embeddings)
- collection.query(query_embeddings, n_results, include)
- collection.update / collection.delete / collection.upsert
"""

import chromadb
from chromadb.utils import embedding_functions


# --- 初始化 ---
# 嵌入式模式：Chromadb 作为 Python 对象存在，随 FastAPI 进程启动/退出
# 不是独立 server，没有 HTTP 调用开销
client = chromadb.PersistentClient(path="./data/chroma_db")

# --- Embedding 函数 ---
# 内部 text-embedding-v3（768 维），走 gRPC 调用
# 生产环境是自定义的 EmbeddingFunction 子类，对接内部 API
# 这里只展示 ChromaDB 侧的接口形态
ef = embedding_functions.DefaultEmbeddingFunction()  # 占位，实际是自定义 gRPC 实现

# --- 创建 Collection ---
collection = client.get_or_create_collection(
    name="docs_v1",
    metadata={"hnsw:space": "cosine"},       # HNSW 索引，余弦相似度
    embedding_function=ef,
)

# --- 写入向量 ---
# 离线阶段每条 chunk 生成 embedding 后批量写入
collection.add(
    ids=["doc1_chunk0", "doc1_chunk1", "doc2_chunk0"],
    documents=[
        "OA 监控告警配置步骤：1. 登录控制台...",
        "告警阈值设置建议：CPU 使用率超过 80%...",
        "Region 白名单配置：进入多AZ部署页面...",
    ],
    metadatas=[
        {"file_name": "oa_monitor.md", "headers": "OA > 监控告警 > 配置步骤", "chunk_index": 0},
        {"file_name": "oa_monitor.md", "headers": "OA > 监控告警 > 阈值设置", "chunk_index": 1},
        {"file_name": "region_whitelist.md", "headers": "Region白名单 > 多AZ部署", "chunk_index": 0},
    ],
    # embeddings 参数不传则自动调 embedding_function 生成
)

# --- 向量检索 ---
# HNSW 图搜索，O(log N)，2000 chunk 下 ~3ms
results = collection.query(
    query_embeddings=[ef(["OA 告警阈值怎么设置"])[0]],  # 先 embedding 再检索
    n_results=5,
    include=["documents", "metadatas", "distances"],
)
# results["ids"][0]       → ["doc1_chunk1", ...]
# results["documents"][0] → ["告警阈值设置建议：CPU...", ...]
# results["metadatas"][0] → [{"file_name": "oa_monitor.md", ...}, ...]
# results["distances"][0] → [0.12, ...]  # 余弦距离，越小越相似


# --- 索引切换（Q37 上线策略）---
# 新策略构建在新 collection，旧的不动，靠配置指针切换
collection_v2 = client.get_or_create_collection(name="docs_v2", embedding_function=ef)
# ... 离线构建新索引 ...
# 切换：active_collection = "docs_v2"，改配置重启即可
# 回滚：active_collection 改回 "docs_v1"，秒级
