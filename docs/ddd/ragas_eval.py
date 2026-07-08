"""
RAGAS 评估 —— LLM-as-judge 自动评测

三个核心指标及其底层计算原理（详见 项目一.md Q15）：

1. Faithfulness（忠实度）—— 答案有没有编造
   计算：拆成原子断言 → 逐条验证是否有上下文依据 → Yes断言数 / 总断言数

2. Answer Relevancy（答案相关性）—— 有没有答非所问
   计算：答案反向生成N个问题 → 每个和原始问题算余弦相似度 → N个相似度的平均值

3. Answer Correctness（答案正确性）—— 事实有没有搞错
   计算：生成答案和 ground truth 拆成陈述 → TP/FP/FN 分类
         → |TP| / (|TP| + 0.5|FP| + 0.5|FN|)

API 要点（面试用）：
- from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
- from ragas import evaluate
- evaluate(dataset, metrics=[...], llm=judge_llm, embeddings=eval_embeddings)
"""

from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
# from langchain_openai import ChatOpenAI  # RAGAS 用 LangChain 封装的 LLM


def build_eval_dataset(qa_pairs: list[dict], retrieval_results: list[dict]) -> Dataset:
    """
    构造 RAGAS 评测数据集

    RAGAS 需要的字段：
    - question: 用户问题
    - answer: LLM 生成的答案
    - contexts: 检索召回的 chunk 列表（Rerank top-5）
    - ground_truth: 标准答案（仅 Answer Correctness 需要，覆盖 100 条 QA）
    """
    return Dataset.from_dict({
        "question":       [qa["question"] for qa in qa_pairs],
        "answer":         [ret["answer"] for ret in retrieval_results],
        "contexts":       [ret["top_chunks"] for ret in retrieval_results],
        "ground_truth":   [qa.get("ground_truth", "") for qa in qa_pairs],
    })


def run_ragas_eval(dataset: Dataset):
    """
    跑 RAGAS 评测

    LLM-as-judge 可靠性保证（Q15）：
    1. 裁判模型和生成模型分离 —— 避免同一个模型既当运动员又当裁判
    2. 结构化输出 —— prompt 要求 JSON 格式，减少裁判自由发挥
    3. 人工抽查校准 —— 每轮随机抽 10~15 条人工复评
       Faithfulness 一致率 ~85%，Answer Relevancy ~80%
    4. 阈值设保守 —— Faithfulness 阈值 0.85 而不是 0.9，留出误差余量
    """
    # judge_llm = ChatOpenAI(model="internal-judge-model", temperature=0)
    # eval_embeddings = CustomEmbeddingFunction()

    # result = evaluate(
    #     dataset=dataset,
    #     metrics=[
    #         faithfulness,           # 忠实度
    #         answer_relevancy,       # 答案相关性
    #         answer_correctness,     # 答案正确性（需 ground_truth）
    #     ],
    #     llm=judge_llm,
    #     embeddings=eval_embeddings,
    # )
    # return result
    pass


# --- 三个指标的底层计算逻辑（面试详解用）---

def faithfulness_explained(answer: str, contexts: list[str]):
    """
    Faithfulness 底层三步：

    Step 1: 裁判 LLM 把答案拆成原子化断言
      answer="OA 告警阈值是 80%，告警后 oncall 需在 5 分钟内响应"
      → claims = ["OA 告警阈值是 80%", "告警后 oncall 需在 5 分钟内响应"]

    Step 2: 逐条断言验证 —— "这条断言能从上下文推断出来吗？"
      claim="OA 告警阈值是 80%" → 查 contexts → 有依据 → Yes
      claim="告警后 oncall 需在 5 分钟内响应" → 查 contexts → 有依据 → Yes

    Step 3: Faithfulness = Yes断言数 / 总断言数 = 2/2 = 1.0
    """
    # RAGAS 实际调用：
    # prompt = f"""
    # Given the following contexts: {contexts}
    # And the answer: {answer}
    # Extract all factual claims from the answer, and for each claim,
    # determine if it can be inferred from the contexts.
    # Output JSON: [{{"claim": "...", "verdict": "Yes/No", "reason": "..."}}]
    # """
    pass


def answer_relevancy_explained(answer: str, original_question: str):
    """
    Answer Relevancy 底层三步：

    Step 1: 裁判 LLM 反向推演 —— 这个答案可能在回答什么问题？
      answer="SMS 通知渠道需在 OA 告警配置中开启"
      → reversed_questions = [
            "如何开启 SMS 通知？",
            "SMS 通知在哪里配置？",
            "OA 告警怎么配置 SMS 通知？"
        ]

    Step 2: 每个反向问题 vs 原始问题，算 embedding 余弦相似度
      cos(emb("如何开启 SMS 通知？"), emb("OA告警怎么配SMS")) = 0.95
      cos(emb("SMS 通知在哪里配置？"), emb("OA告警怎么配SMS")) = 0.91
      cos(emb("OA 告警怎么配置 SMS 通知？"), emb("OA告警怎么配SMS")) = 0.97

    Step 3: Answer Relevancy = (0.95 + 0.91 + 0.97) / 3 = 0.943

    如果答案跑题了（大谈监控大盘历史），反向问题会和原始问题偏离很远 → 低分
    """
    pass


def answer_correctness_explained(answer: str, ground_truth: str):
    """
    Answer Correctness 底层三步（需 ground truth）：

    Step 1: 裁判 LLM 把生成答案和 ground truth 各自拆成原子化陈述
      answer → ["告警阈值 80%", "响应时间 5min", "配置在 OA 控制台"]
      ground_truth → ["告警阈值 80%", "响应时间 10min", "配置在 OA 控制台", "需开启 SMS"]

    Step 2: 逐条比对分类
      TP: ["告警阈值 80%", "配置在 OA 控制台"]           → 2 个
      FP: ["响应时间 5min"]  （答案说5min，实际是10min）  → 1 个
      FN: ["响应时间 10min", "需开启 SMS"]  （答案漏了）   → 2 个

    Step 3: Correctness = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
            = 2 / (2 + 0.5×1 + 0.5×2)
            = 2 / 3.5
            = 0.571
    0.5 权重：FP 和 FN 各算半个惩罚，漏说和说错都扣分，但不至于一票否决
    """
    pass


# --- 指标阈值（Q15 + Q23）---
EVAL_THRESHOLDS = {
    "Faithfulness":         {"threshold": 0.85, "说明": "低于此值排查幻觉"},
    "Answer Relevancy":     {"threshold": None, "说明": "没有硬阈值，看趋势"},
    "Answer Correctness":   {"threshold": None, "说明": "只覆盖100条有ground truth的QA"},
    "人工-RAGAS一致率":      {"Faithfulness": "~85%", "Answer Relevancy": "~80%"},
}
