"""
RAGAS 评测模拟 —— 演示 Faithfulness / Answer Relevancy / Answer Correctness 三个核心指标的 API 用法

不依赖真实 RAGAS 安装，数据全部 mock。安装 ragas 后直接跑：pip install ragas datasets
"""

import json
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# 1. 模拟 RAGAS 所需的数据结构
# ============================================================

@dataclass
class Sample:
    """RAGAS 评测需要的最小数据单元"""
    question: str            # 用户原始问题
    answer: str              # LLM 最终生成的答案
    contexts: list[str]      # 检索 + Rerank 后喂给 LLM 的 chunk 列表
    ground_truth: str        # 正确答案（仅 Answer Correctness 用到）


# ============================================================
# 2. 模拟评测数据集 —— 你的 100 条 QA pairs
# ============================================================

def build_mock_eval_dataset() -> list[Sample]:
    """
    模拟数据，覆盖三种场景：
      - 正常回答（Faithfulness 高）
      - 幻觉回答（Faithfulness 低，编了 context 里没有的信息）
      - 答非所问（Answer Relevancy 低）
    """
    return [
        Sample(
            question="OA 的告警阈值怎么配置？",
            answer="OA 告警阈值在 config/alarm.yaml 中配置，支持 CPU 超过 80% 和内存超过 90% 两个维度。",
            contexts=[
                "OA 监控告警配置文件为 config/alarm.yaml，默认开启 CPU 和内存告警。",
                "CPU 告警阈值为 80%，内存告警阈值为 90%，修改后需重启 OA 服务生效。",
                "OA 是优化顾问（Optimization Advisor）的缩写，用于云资源监控。",
            ],
            ground_truth="OA 告警阈值在 config/alarm.yaml 中配置，支持 CPU 超 80% 和内存超 90%。",
        ),
        Sample(
            question="region 白名单在哪里配？",
            answer="region 白名单在 settings/region_whitelist.json 中配置，支持按 region ID 粒度的访问控制。",
            contexts=[
                "region 白名单配置文件为 settings/region_whitelist.json，格式为 JSON 数组。",
                "每个 region 通过 region_id 字段标识，白名单控制该 region 的 API 访问权限。",
            ],
            ground_truth="region 白名单在 settings/region_whitelist.json 中按 region ID 配置。",
        ),
        Sample(
            question="Redis 分布式锁的 TTL 是多少？",
            # 故意编造：context 里没有 "30 秒" 这个信息
            answer="Redis 分布式锁的 TTL 是 30 秒，超时自动释放。",
            contexts=[
                "文档更新时使用 Redis 分布式锁做 file_name 级别的并发控制。",
                "锁的 key 格式为 doc_lock:{file_name}。",
            ],
            ground_truth="锁 TTL 为 30 秒。",
        ),
        Sample(
            question="embedding 模型用的什么？",
            # 答非所问：问的是模型名，回答讲的是切片策略
            answer="我们的切片策略是按 Markdown 标题层级做语义切分，384 token 一个 chunk。",
            contexts=[
                "使用阿里云 text-embedding-v3 模型做向量化，输出 768 维，L2 归一化。",
                "chunk size 选了 384 token，截断率约 1%。",
            ],
            ground_truth="使用阿里云 text-embedding-v3 模型。",
        ),
    ]


# ============================================================
# 3. 模拟 RAGAS 评测管线（如果装了 ragas，直接替换为真实调用）
# ============================================================

def run_ragas_evaluation(samples: list[Sample], llm_for_judge: str = "deepseek-v4-flash"):
    """
    RAGAS 评测主函数。

    真实 RAGAS 调用方式（伪代码见注释），这里用模拟打分代替，方便不装依赖直接跑。
    """
    results = []
    for i, s in enumerate(samples):
        # ---- 真实 RAGAS 调用 ----
        # from ragas import evaluate
        # from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
        # from datasets import Dataset
        #
        # ds = Dataset.from_dict({
        #     "question": [s.question],
        #     "answer": [s.answer],
        #     "contexts": [s.contexts],
        #     "ground_truth": [s.ground_truth],
        # })
        #
        # score = evaluate(
        #     ds,
        #     metrics=[faithfulness, answer_relevancy, answer_correctness],
        #     llm=llm_for_judge,        # 裁判模型（要和生成模型分离）
        #     embeddings=embedding_model, # 用于 answer_relevancy 的语义相似度计算
        # )

        # ---- 模拟打分（用规则替代 LLM 判断） ----
        faithfulness = _mock_faithfulness(s)
        relevancy = _mock_answer_relevancy(s)
        correctness = _mock_answer_correctness(s)

        results.append({
            "id": i + 1,
            "question": s.question[:40] + "...",
            "faithfulness": round(faithfulness, 4),
            "answer_relevancy": round(relevancy, 4),
            "answer_correctness": round(correctness, 4),
            "pass": faithfulness >= 0.85,
        })

    return results


# ============================================================
# 4. 三个核心指标的模拟实现（实际是 RAGAS 内部逻辑的简化版）
# ============================================================

def _mock_faithfulness(s: Sample) -> float:
    """
    Faithfulness（忠实度）

    RAGAS 真实流程：
      1. 用 LLM 把 answer 拆成原子化断言列表：["断言1", "断言2", ...]
      2. 对每条断言，用 LLM 判断是否能从 contexts 中找到依据
      3. Faithfulness = 有依据的断言数 / 总断言数

    模拟：检查 answer 中的数字/专有名词是否在 contexts 中出现
    """
    import re

    claims = _extract_claims(s.answer)
    if not claims:
        return 0.0

    verified = 0
    context_text = " ".join(s.contexts)
    for claim in claims:
        # 检查断言中的关键实体是否在 context 中有支撑
        if _claim_supported(claim, context_text):
            verified += 1

    return verified / len(claims)


def _mock_answer_relevancy(s: Sample) -> float:
    """
    Answer Relevancy（答案相关性）

    RAGAS 真实流程：
      1. 对 answer 反向生成 N 个问题（"这段回答能回答什么问题？"）
      2. 计算每个生成问题与原始 user question 的 embedding 余弦相似度
      3. Answer Relevancy = 所有相似度的平均值

    模拟：检查 answer 和 question 的关键词重叠度
    """
    q_words = set(_tokenize(s.question))
    a_words = set(_tokenize(s.answer))

    if not q_words:
        return 0.0

    overlap = q_words & a_words
    return len(overlap) / len(q_words)


def _mock_answer_correctness(s: Sample) -> float:
    """
    Answer Correctness（答案正确性）

    RAGAS 真实流程：
      1. 用 LLM 对比 answer 和 ground_truth，提取事实点
      2. 计算 TP（answer 说对的事实）、FP（answer 说错的事实）、FN（ground_truth 有但 answer 漏了的事实）
      3. F1 = 2 * TP / (2*TP + FP + FN)，即 correctness 分数

    模拟：关键词 + 关键数字的精确匹配
    """
    gt_words = set(_tokenize(s.ground_truth))
    ans_words = set(_tokenize(s.answer))

    if not gt_words:
        return 0.0

    tp = len(gt_words & ans_words)       # 答对的
    fp = len(ans_words - gt_words)       # 答错的/多余的
    fn = len(gt_words - ans_words)       # 漏掉的

    if 2 * tp + fp + fn == 0:
        return 0.0
    return (2 * tp) / (2 * tp + fp + fn)  # F1 公式


# ============================================================
# 5. 辅助函数
# ============================================================

def _extract_claims(text: str) -> list[str]:
    """按句号/分号切分，模拟 RAGAS 的断言拆解"""
    import re
    parts = re.split(r"[。；;]", text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def _claim_supported(claim: str, context: str) -> bool:
    """检查断言是否在 context 中有支撑（简化版：关键词匹配）"""
    words = _tokenize(claim)
    if not words:
        return False
    matched = sum(1 for w in words if w in context)
    return matched / len(words) >= 0.5


def _tokenize(text: str) -> list[str]:
    """简易中文分词（实际项目用 jieba）"""
    import re
    # 提取中文词 + 英文词 + 数字
    tokens = re.findall(r"[一-鿿]+|[a-zA-Z]+|\d+", text.lower())
    # 过滤停用词
    stopwords = {"的", "是", "在", "了", "和", "与", "或", "不", "都", "也", "到", "把", "被", "从", "对", "等", "及", "之", "为", "以"}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


# ============================================================
# 6. 完整的 RAGAS 真实调用示例（注释形式，供参考）
# ============================================================

"""
# ===== 真实 RAGAS 安装和调用 =====

# pip install ragas datasets langchain-openai

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1. 准备裁判 LLM（必须和生成 LLM 分开）
judge_llm = LangchainLLMWrapper(ChatOpenAI(
    model="deepseek-v4-flash",
    api_key="<your-api-key>",
    base_url="<internal-api-url>",
))

# 2. 准备 embedding 模型（answer_relevancy 需要）
embed_model = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
    model="text-embedding-v3",
    api_key="<your-api-key>",
    base_url="<internal-api-url>",
))

# 3. 构建数据集
eval_data = {
    "question": [s.question for s in samples],
    "answer": [s.answer for s in samples],
    "contexts": [s.contexts for s in samples],      # list[list[str]]
    "ground_truth": [s.ground_truth for s in samples],
}
dataset = Dataset.from_dict(eval_data)

# 4. 运行评测
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, answer_correctness],
    llm=judge_llm,
    embeddings=embed_model,
)

# 5. 结果导出
df = result.to_pandas()
print(df[["faithfulness", "answer_relevancy", "answer_correctness"]].describe())

# 6. 按阈值筛选 bad case
bad_cases = df[df["faithfulness"] < 0.85]
print(f"Faithfulness < 0.85 的 bad case 数量: {len(bad_cases)}")
for _, row in bad_cases.iterrows():
    print(f"  Q: {row['question'][:50]}")
    print(f"  Faithfulness: {row['faithfulness']:.3f}")
    print()
"""


# ============================================================
# 7. 主程序
# ============================================================

def main():
    print("=" * 60)
    print("RAGAS 评测模拟 —— 100 条 QA pairs 场景")
    print("=" * 60)

    samples = build_mock_eval_dataset()
    print(f"\n评测集规模: {len(samples)} 条（实际线上 100 条）\n")

    results = run_ragas_evaluation(samples)

    # 打印每条结果
    print(f"{'ID':<4} {'Question':<42} {'Faith':<8} {'Relev':<8} {'Correct':<8} {'Pass'}")
    print("-" * 80)
    for r in results:
        flag = "PASS" if r["pass"] else "FAIL (幻觉/答非所问)"
        print(f"{r['id']:<4} {r['question']:<42} {r['faithfulness']:<8} {r['answer_relevancy']:<8} {r['answer_correctness']:<8} {flag}")

    # 汇总
    faith_avg = sum(r["faithfulness"] for r in results) / len(results)
    relev_avg = sum(r["answer_relevancy"] for r in results) / len(results)
    correct_avg = sum(r["answer_correctness"] for r in results) / len(results)
    pass_rate = sum(1 for r in results if r["pass"]) / len(results)

    print("\n" + "=" * 60)
    print("汇总报告（模拟 RAGAS 输出）")
    print("=" * 60)
    print(f"  Faithfulness      均值: {faith_avg:.4f}  (阈值 0.85)")
    print(f"  Answer Relevancy  均值: {relev_avg:.4f}")
    print(f"  Answer Correctness 均值: {correct_avg:.4f}")
    print(f"  通过率 (Faith>=0.85): {pass_rate:.1%}  ({sum(1 for r in results if r['pass'])}/{len(results)})")

    # 人工抽查提示
    print(f"\n人工抽查: 随机抽 {max(1, len(results)//3)} 条做 Faithfulness 人工复评")
    print("对比 RAGAS 分数和人工判断 → 计算一致率")


if __name__ == "__main__":
    main()
