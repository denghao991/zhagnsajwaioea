"""
FastAPI /ask 端点 —— 全链路编排

核心认知：
- FastAPI + Uvicorn，4 个实例 + Nginx 轮询
- /ask 是主端点，分两条路径：缓存命中快速通道 / 缓存未命中完整检索
- SSE 流式输出（StreamingResponse）：适合 I/O 密集型链路（频繁调内部 API 等待响应）
- 每个请求有 session_id 追踪，带 user_id 关联日志
- 无状态设计 —— 每次 /ask 独立，不依赖之前会话

API 要点（面试用）：
- @app.post("/ask") → 接收 question + user_id
- StreamingResponse → SSE 推事件
- 每条 yield 是一个 SSE event：rewrite / context / token / done
"""

import asyncio
import uuid
from starlette.responses import StreamingResponse


# --- /ask 端点（全链路）---
async def handle_ask(request: dict) -> StreamingResponse:
    """
    主请求处理 —— 缓存命中快速通道 / 缓存未命中完整检索

    输入：{"question": "OA 告警阈值怎么设置", "user_id": "zhangsan"}
    输出：SSE 事件流
    """

    async def event_stream():
        question = request["question"]
        user_id = request["user_id"]
        session_id = str(uuid.uuid4())
        t0 = asyncio.get_event_loop().time()

        try:
            # ── Step 1: 语义缓存检查（Q21）──
            query_embedding = await embed_query(question)
            cached_answer = semantic_cache.lookup(query_embedding)
            if cached_answer:
                yield sse("answer", {"text": cached_answer, "cache": "hit"})
                yield sse("done", {"cache": "hit"})
                # 缓存命中也写 QueryLog
                write_querylog(user_id, session_id, question, question, [], cached_answer,
                              (asyncio.get_event_loop().time() - t0) * 1000, cache_hit=True)
                return

            # ── Step 2: Query 改写（Q17/Q18，~100ms）──
            rewritten = rewrite_pipeline(question)
            yield sse("rewrite", {"original": question, "rewritten": rewritten})

            # ── Step 3: Embedding（~30ms，gRPC 调用）──
            query_embedding = await embed_query(rewritten)

            # ── Step 4: 双路并行检索（~5ms）──
            hybrid_results = hybrid_search(
                rewritten, query_embedding, collection, bm25, chunks
            )
            yield sse("retrieval", {
                "candidates": len(hybrid_results),
                "sources": {
                    "vector": sum(1 for r in hybrid_results if r.get("source") == "vector"),
                    "bm25":   sum(1 for r in hybrid_results if r.get("source") == "bm25"),
                    "both":   sum(1 for r in hybrid_results if r.get("source") == "both"),
                }
            })

            # ── Step 5: Rerank（~100ms）──
            top_chunks = rerank(rewritten, hybrid_results, top_k=5)
            yield sse("context", {
                "chunks": [
                    {"file": r["metadata"]["file_name"], "preview": r["content"][:120]}
                    for r in top_chunks
                ]
            })

            # ── Step 6: LLM 生成（流式，P50 ~400ms）──
            full_answer = ""
            async for token in llm_stream_generate(question, top_chunks):
                full_answer += token
                yield sse("token", {"text": token})

            # ── Step 7: 写 QueryLog（<1ms）──
            latency_ms = (asyncio.get_event_loop().time() - t0) * 1000
            write_querylog(user_id, session_id, question, rewritten,
                          top_chunks, full_answer, latency_ms, cache_hit=False)

            yield sse("done", {"cache": "miss"})

        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── SSE 事件格式化 ──
def sse(event: str, data: dict) -> str:
    import json
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── 管理端点 ──
async def admin_reload_jieba():
    """POST /admin/jieba/reload —— 热加载 jieba 词库"""
    from jieba_dict import reload_jieba_dict
    return reload_jieba_dict()


async def admin_rebuild_index():
    """POST /admin/rebuild —— 触发离线索引重建（Q26，收归单人操作）"""
    # 新建 collection docs_v2 → 重新切分 → embedding → 写入
    # 完成后切 active_collection 配置指针
    pass


# ── 健康检查 ──
async def health_check():
    """Nginx 用健康检查判断节点是否可用（Q27）"""
    # 检查 ChromaDB 是否可读 + BM25 倒排索引是否已加载
    # chunk 总数比对：和重建完成的节点一致则通过
    return {"status": "ok", "chunks": current_chunk_count}


# ── 延迟汇总（面试用，Q24 数据）──
LATENCY_BUDGET = {
    "步骤":          ["Query改写", "Embedding", "双路检索", "RRF融合", "Rerank", "QueryLog", "检索段小计", "LLM生成", "端到端"],
    "P50":           ["100ms",    "30ms",     "5ms",     "<1ms",   "100ms", "<1ms",     "~235ms",  "400ms", "~635ms"],
    "P99":           ["500ms",    "60ms",     "15ms",    "<1ms",   "250ms", "<1ms",     "~825ms",  "2.5s",  "~3.3s"],
    "操作":           ["gRPC",     "gRPC",     "内存",    "内存",   "gRPC",  "SQLite",   "",        "gRPC",  ""],
}
