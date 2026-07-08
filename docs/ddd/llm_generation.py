"""
LLM 生成 —— DeepSeek V4 Flash + Prompt Template + SSE 流式输出

核心认知：
- 模型走内部统一 API（gRPC），不是自己部署的
- System prompt + Few-shot + Context + Query → Prompt template
- 每个 chunk 附带文件来源标注，LLM 回答中引用 `[1]` `[2]`
- SSE 流式输出：首 token 到达即开始渲染，用户感知延迟远低于完整生成时间
- 生成阶段是整个链路延迟最大的部分：P50 400ms, P99 2.5s

API 要点（面试用）：
- StreamingResponse → SSE 逐 token 推送
- Prompt 模板拼接：System + Few-shot + Context + Query
- 温度设 0 降低幻觉
"""


# --- Prompt 模板 ---
SYSTEM_PROMPT = """你是内部知识库助手，只使用参考资料中明确包含的信息回答问题。
规则：
1. 如果你的知识库文档中有明确答案，请直接回答并标注来源编号 [1] [2]
2. 如果文档中没有相关内容，直接告知用户"文档中未找到相关信息"，不要编造
3. 不要推测、不要使用外部知识、不要补充文档中没有的细节
4. 每个关键陈述后标注来源编号"""

FEW_SHOT_EXAMPLE = """
参考资料：
[1] OA监控告警配置步骤：1. 登录控制台 2. 进入OA管理页面...
[2] 告警阈值设置：CPU使用率超过80%触发告警，内存超过90%...

问题：OA告警阈值怎么设置？
回答：根据文档，OA监控告警的阈值设置为 CPU 使用率超过 80% 触发告警 [2]，内存使用率超过 90% 触发告警 [2]。配置路径为控制台 → OA管理页面 → 告警设置 [1]。
"""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    拼接完整 Prompt：System + Few-shot + Context + Query
    输入约 2750 token（含 5 个 chunk × ~350 token/chunk）
    """
    # 拼接上下文（每个 chunk 带来源标注）
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = f"{chunk['metadata']['file_name']} > {chunk['metadata'].get('headers', '')}"
        context_parts.append(f"[{i}] {chunk['content']}  （来源：{source}）")
    context = "\n\n".join(context_parts)

    return f"""{SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLE}

参考资料：
{context}

问题：{query}
回答："""


# --- LLM 生成（流式 SSE）---
async def llm_stream_generate(query: str, chunks: list[dict]):
    """
    流式生成 —— DeepSeek V4 Flash，约 30~50 token/s

    SSE 事件格式：
    data: {"token": "OA"}
    data: {"token": "监控"}
    ...
    data: {"done": true}
    """
    prompt = build_prompt(query, chunks)

    # 实际走内部 gRPC streaming
    # async for response in llm_client.stream_generate(
    #     model="deepseek-v4-flash",
    #     prompt=prompt,
    #     max_tokens=500,
    #     temperature=0,  # 确定性输出，降低幻觉
    # ):
    #     yield response.token
    pass


# --- 延迟预算 ---
def latency_budget():
    """完整生成阶段的延迟分布（Q24）"""
    return {
        "P50": {
            "query_rewrite":    "100ms",
            "embedding":        "30ms",
            "dual_search":      "5ms  (HNSW 3ms + BM25 2ms, 并行取max)",
            "rrf_fusion":       "<1ms (纯内存计算)",
            "rerank":           "100ms (8 candidates × cross-encoder)",
            "querylog_write":   "<1ms (SQLite 本地)",
            "检索段小计":        "~235ms",
            "llm_generation":   "400ms (200 token / ~50 tok/s)",
            "端到端总计":        "~635ms",
        },
        "P99": {
            "检索段小计":        "~825ms",
            "llm_generation":   "2500ms (500 token + GPU排队)",
            "端到端总计":        "~3300ms",
        },
        "LLM生成占总延迟":       "63% (P50), 是端到端延迟的绝对大头",
        "缓存命中":              "< 100ms (跳过改写/检索/Rerank/生成全部步骤)",
    }
