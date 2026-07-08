"""
RAG 系统所有 Prompt 模板 —— 可直接复制使用

包含：
1.  System Prompt（LLM 生成）
2.  Few-shot 示例（LLM 生成）
3.  Query 改写 Prompt
4.  RAGAS Faithfulness 评测 Prompt
5.  RAGAS Answer Relevancy 评测 Prompt
6.  RAGAS Answer Correctness 评测 Prompt
7.  Rerank 输入格式
8.  特殊文本摘要 Prompt（表格/代码块/JSON）
9.  拒答 Prompt（检索质量兜底）
10. 引用编号要求（拼入 System Prompt）
"""

# ═══════════════════════════════════════════════════════════════
# 1. System Prompt —— LLM 生成（Q23 第二层防线）
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是内部知识库助手。你的知识来源仅限于下方「参考资料」中提供的文档内容。

规则：
1. 只使用参考资料中明确包含的信息来回答问题。不允许使用外部知识、常识推测或训练数据中的记忆。
2. 如果参考资料中有明确答案，请直接回答，并在每个关键陈述后用 [数字] 标注来源编号。
3. 如果参考资料中没有相关内容，直接告知用户"文档中未找到相关信息"，不要编造、不要推测、不要补充。
4. 回答要简洁完整，不要展开与问题无关的背景介绍。"""


# ═══════════════════════════════════════════════════════════════
# 2. Few-shot 示例 —— 拼在 System Prompt 和 Context 之间
# ═══════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLE = """
【示例】

参考资料：
[1] OA监控告警配置步骤：1. 登录控制台 2. 进入OA管理页面 3. 选择告警配置...
[2] 告警阈值设置建议：CPU使用率超过80%触发告警，内存使用率超过90%触发告警。告警后系统自动发送SMS通知给oncall人员。
[3] OA监控告警支持的通知渠道：SMS、邮件、企业微信。SMS通知需在渠道配置中开启。

问题：OA告警阈值怎么设置？支持哪些通知方式？
回答：OA监控告警的阈值设置为 CPU 使用率超过 80%、内存使用率超过 90% 时触发告警 [2]。告警触发后，系统支持 SMS、邮件、企业微信三种通知方式 [3]，其中 SMS 通知需要在渠道配置中手动开启 [3]。配置路径为：控制台 → OA管理页面 → 告警配置 [1]。
"""


# ═══════════════════════════════════════════════════════════════
# 3. Query 改写 Prompt（Q17/Q18 —— 第二层 LLM 去口语化）
# ═══════════════════════════════════════════════════════════════

REWRITE_SYSTEM_PROMPT = """你是一个查询改写器。你的任务是将用户的口语化技术问题改写为正式、精确的文档查询语句。

规则：
1. 保留原文中的所有数字、否定词（不、没、无等）、时间表达 —— 不要修改或删除
2. 保留所有专有名词和缩写，不要翻译或展开（如"OA""AZ""SMS""多AZ"等不做任何改动）
3. 只做口语→正式的转换：将闲聊式表达改为陈述式，将指代不明的词具体化
4. 不要添加用户没问的内容，不要假设用户的意图
5. 输出只包含改写后的单个查询语句，不要任何解释、不要前缀"""

REWRITE_EXAMPLE_USER = "手机收不到短信怎么办"
REWRITE_EXAMPLE_ASSISTANT = "SMS通知渠道未收到告警"


def build_rewrite_prompt(original_query: str) -> str:
    """构建改写 Prompt —— 拼接 system + few-shot + user query"""
    return f"""{REWRITE_SYSTEM_PROMPT}

示例：
用户输入：{REWRITE_EXAMPLE_USER}
改写结果：{REWRITE_EXAMPLE_ASSISTANT}

用户输入：{original_query}
改写结果："""


# ═══════════════════════════════════════════════════════════════
# 4. RAGAS Faithfulness 评测 Prompt（Q15）
#    裁判 LLM 拆断言 → 逐条验证 → 算分
# ═══════════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """Your task is to judge the faithfulness of an answer given a set of contexts.

First, extract ALL factual claims from the answer. A claim is a single, independently verifiable statement of fact. Break compound sentences into atomic claims.

Example:
Answer: "OA 告警阈值是 80%，告警后 oncall 需在 5 分钟内响应。"
Claims:
1. OA 告警阈值是 80%
2. 告警后 oncall 需在 5 分钟内响应

Second, for each claim, determine whether it can be inferred from the provided contexts. A claim is "inferable" if the context directly states it or the information can be logically deduced from the context.

Output in JSON format:
{{
    "claims": [
        {{"claim": "OA 告警阈值是 80%", "verdict": "Yes", "reason": "上下文明确提到 CPU 使用率超过 80% 触发告警"}},
        {{"claim": "告警后 oncall 需在 5 分钟内响应", "verdict": "No", "reason": "上下文中没有提到 5 分钟响应时间"}}
    ],
    "score": 0.5
}}

Contexts:
{contexts}

Answer:
{answer}"""


# ═══════════════════════════════════════════════════════════════
# 5. RAGAS Answer Relevancy 评测 Prompt（Q15）
#    答案反向生成问题 → 和原始问题算语义相似度
# ═══════════════════════════════════════════════════════════════

ANSWER_RELEVANCY_PROMPT = """Your task is to generate questions that the given answer could be responding to.

Given the answer below, generate 3 questions that this answer would be an appropriate response to. The questions should cover different aspects of the answer. Output in JSON format.

Answer:
{answer}

Output format:
{{
    "questions": [
        "question 1",
        "question 2",
        "question 3"
    ]
}}"""


# ═══════════════════════════════════════════════════════════════
# 6. RAGAS Answer Correctness 评测 Prompt（Q15）
#    答案 vs ground truth → TP/FP/FN 分类 → F1-style
# ═══════════════════════════════════════════════════════════════

ANSWER_CORRECTNESS_PROMPT = """Your task is to evaluate the correctness of a generated answer against a ground truth answer.

Step 1: Extract all atomic factual statements from both the generated answer and the ground truth.

Step 2: Classify each statement from the generated answer:
- TP (True Positive): appears in both generated answer and ground truth
- FP (False Positive): appears in generated answer but NOT in ground truth (incorrect or fabricated)
- FN (False Negative): appears in ground truth but NOT in generated answer (missing information)

Step 3: Calculate score = |TP| / (|TP| + 0.5 * |FP| + 0.5 * |FN|)

Output in JSON format:
{{
    "tp_statements": ["statement 1", "statement 2"],
    "fp_statements": ["incorrect statement"],
    "fn_statements": ["missing statement 1", "missing statement 2"],
    "score": 0.75,
    "explanation": "简要说明扣分原因"
}}

Ground truth:
{ground_truth}

Generated answer:
{answer}"""


# ═══════════════════════════════════════════════════════════════
# 7. Rerank 输入格式（Q11）
#    Cross-encoder 将 query 和 doc 拼接后做全注意力推理
# ═══════════════════════════════════════════════════════════════

def build_rerank_input(query: str, document: str) -> str:
    """Qwen3-rerank 的输入格式：query + document 拼接"""
    return f"<query>{query}</query>\n<document>{document}</document>"


# 批量 Rerank 调用格式（gRPC 调用形态）
# rerank_client.rerank(
#     model="qwen3-rerank",
#     query=query,
#     documents=[doc1_content, doc2_content, ..., doc8_content],
#     top_n=5,
# )
# 返回每个 document 的相关性分数 (float, 0~1)


# ═══════════════════════════════════════════════════════════════
# 8. 特殊文本摘要 Prompt（Q1 —— 表格/代码块/JSON 摘要）
#    让 LLM 为原始数据生成自然语言描述，辅助向量检索
# ═══════════════════════════════════════════════════════════════

SPECIAL_TEXT_SUMMARY_PROMPT = """你是一个技术文档摘要生成器。用一句简洁的中文，描述以下技术内容的核心信息和用途。

规则：
1. 只描述"这是什么、用来做什么"，不要评价
2. 保留关键参数名和数值
3. 一句话，不超过 50 个字

内容：
{special_text}

一句话摘要："""


# ═══════════════════════════════════════════════════════════════
# 9. 拒答 Prompt（Q18 —— 检索质量兜底，当前未启用）
#    在 LLM 生成前判断：如果检索结果质量太低，不进入生成
# ═══════════════════════════════════════════════════════════════

REJECTION_MESSAGE = "抱歉，当前知识库中没有找到相关信息。建议您查看 Wiki 文档或联系人工支持。"

# 拒答判断逻辑（非 Prompt，是代码逻辑）：
# if rerank_top1_score < REJECTION_THRESHOLD:
#     且 vector ∩ bm25 交集为空:
#     → 返回 REJECTION_MESSAGE，不进入 LLM 生成

# 阈值确定方法（Q18）：
# 拉一批「知识库确实没有答案」的 query，跑一遍完整检索，
# 画出 rerank top-1 分数分布，取能区分「有答案」和「没答案」的最优切分点


# ═══════════════════════════════════════════════════════════════
# 10. 引用编号要求 —— 拼入 System Prompt 的结构性约束
# ═══════════════════════════════════════════════════════════════

CITATION_REQUIREMENT = """引用规则：
- 参考资料中的每条 chunk 有编号 [1] [2] [3]...
- 回答中每个关键事实陈述后必须标注来源编号，例如"CPU 阈值设为 80% [2]"
- 如果一个陈述综合了多个来源，标注所有相关编号，例如"... [1][3]"
- 如果某个陈述是你自己的总结而非来自具体某条资料，不要标注编号
- 编号必须对应到上面参考资料中的实际编号，不能编造不存在的编号"""


# ═══════════════════════════════════════════════════════════════
# 附录：完整 Prompt 拼接模板（面试时可直接讲）
# ═══════════════════════════════════════════════════════════════

def build_full_generation_prompt(query: str, chunks: list[dict]) -> str:
    """
    完整 Prompt 拼接顺序（Q17 第六步 + Q23 第二层）:

    ┌──────────────────────┐
    │  System Prompt       │  ← 行为约束："只使用资料中的信息"
    │  + Citation Rule     │  ← 结构性约束："每句话标注来源"
    ├──────────────────────┤
    │  Few-shot Example    │  ← 输出模式示范："精确回答、带来源、不推测"
    ├──────────────────────┤
    │  Contexts            │  ← Rerank top-5 chunk，每个带编号 [1]~[5]
    │  (每个chunk:         │
    │   [N] content        │
    │   来源：file > path) │
    ├──────────────────────┤
    │  User Query          │  ← 用户原始问题（非改写后的，保证用户看懂）
    └──────────────────────┘

    总计约 2750 token（5 chunk × ~350 token + system ~200 + few-shot ~400 + query ~50）
    """
    # 拼接上下文
    contexts = []
    for i, chunk in enumerate(chunks, start=1):
        source = f"{chunk['metadata']['file_name']}"
        headers = chunk['metadata'].get('headers', '')
        if headers:
            source += f" > {headers}"
        contexts.append(f"[{i}] {chunk['content']}\n   （来源：{source}）")

    context_text = "\n\n".join(contexts)

    return f"""{SYSTEM_PROMPT}

{CITATION_REQUIREMENT}

{FEW_SHOT_EXAMPLE}

参考资料：
{context_text}

问题：{query}
回答："""
