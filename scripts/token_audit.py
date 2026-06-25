"""全链路 Token 用量审计脚本。

测量 RAG 管道每个阶段的 token 消耗和费用估算。
运行：PYTHONPATH=src python scripts/token_audit.py
"""

from pathlib import Path

from src.tiny_rag.ingestion.tokenizer import count_tokens, encode
from src.tiny_rag.ingestion.chunker import MarkdownChunker
from src.tiny_rag.config import settings

# ============================================================
# 费用单价（参考公开定价，元/1M tokens）
# ============================================================
PRICING = {
    # DeepSeek
    "deepseek-chat":        {"input": 2.0,  "output": 8.0},
    "deepseek-reasoner":    {"input": 4.0,  "output": 16.0},
    # GLM
    "glm-4-plus":           {"input": 10.0, "output": 10.0},
    # 智谱 Embedding
    "text-embedding-v2":    {"per_1k_chars": 0.005},  # 按字符收费
}

print(f"{'='*70}")
print(f"  Token 全链路审计")
print(f"  模型: {settings.llm_model}")
print(f"  Chunk: size={settings.chunk_size}, overlap={settings.chunk_overlap}")
print(f"{'='*70}")

# ============================================================
# 1. 采样文档
# ============================================================
print(f"\n{'─'*70}")
print("  1. 文档采样 — 上传阶段")
print(f"{'─'*70}")

# 找一份已有文档或生成示例
sample_dir = Path("data")
sample_files = list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.md"))
if sample_files:
    doc_text = sample_files[0].read_text(encoding="utf-8")
    print(f"  文档: {sample_files[0].name}")
else:
    # 用分块配置的典型值估算
    doc_text = "这是业务专有名词测试文档。" * 5000

doc_tokens = count_tokens(doc_text)
doc_chars = len(doc_text)
print(f"  字符数:      {doc_chars:>10,}")
print(f"  预估 Token:  {doc_tokens:>10,}")

# ============================================================
# 2. 分块阶段
# ============================================================
print(f"\n{'─'*70}")
print(f"  2. 分块阶段")
print(f"{'─'*70}")

chunker = MarkdownChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
chunks = chunker.chunk_text(doc_text)
n = len(chunks)
chunk_tokens = [c.token_count for c in chunks]

print(f"  分块数量:    {n:>10}")
print(f"  Token 分布:")
print(f"    - 平均:    {sum(chunk_tokens)/n:>10.1f} / chunk")
print(f"    - 最小:    {min(chunk_tokens):>10} / chunk")
print(f"    - 最大:    {max(chunk_tokens):>10} / chunk")
print(f"    - 总 token: {sum(chunk_tokens):>10,}")

# Embedding API 按字符收费
embed_chars = sum(len(c.text) for c in chunks)
embed_cost = embed_chars / 1000 * PRICING["text-embedding-v2"]["per_1k_chars"]
print(f"\n  嵌入费用（按字符）:")
print(f"    总字符:    {embed_chars:>10,}")
print(f"    单价:      {PRICING['text-embedding-v2']['per_1k_chars']:>10.3f} 元/1k字符")
print(f"    嵌入费用:  RMB {embed_cost:>10.6f}")

total_embed_4chars = embed_chars / 1000 * 4 * PRICING["text-embedding-v2"]["per_1k_chars"]
print(f"    写入 ChromaDB: RMB {total_embed_4chars:.6f}（仅首次）")

# ============================================================
# 3. 检索阶段
# ============================================================
print(f"\n{'─'*70}")
print(f"  3. 检索 + LLM 推理（每次提问）")
print(f"{'─'*70}")

# 问题采样
sample_question = "公司最近的业务指标有什么变化？"
q_tokens = count_tokens(sample_question)

# 检索（取 n_results 个 chunk）
n_results = 5
retrieved_chunks = chunks[:min(n_results, len(chunks))]
context_text = "\n\n".join(c.text for c in retrieved_chunks)
context_tokens = count_tokens(context_text)

print(f"  问题:        \"{sample_question}\"")
print(f"  问题 Token:  {q_tokens:>10}")
print(f"  召回 Chunks: {n_results}")
print(f"  Context 总 Token: {context_tokens:>10,}")

# ============================================================
# 4. LLM 调用
# ============================================================
system_prompt = """你是一个基于文档内容回答问题的助手。
请根据以下提供的文档片段来回应用户的问题。
如果你在文档中找不到相关信息，请诚实地说明你不知道。
不要编造信息。"""
system_tokens = count_tokens(system_prompt)

user_template = f"文档内容：\n{context_text}\n\n问题：{sample_question}"
user_tokens = count_tokens(user_template)

input_tokens = system_tokens + user_tokens

# 估计输出（max_tokens=1024）
output_tokens = 1024

model = settings.llm_model
pricing = PRICING.get(model, PRICING["deepseek-chat"])

input_cost = input_tokens / 1_000_000 * pricing["input"]
output_cost = output_tokens / 1_000_000 * pricing["output"]
total_cost = input_cost + output_cost

print(f"\n  LLM 输入分解:")
print(f"    - System prompt:  {system_tokens:>10} tokens")
print(f"    - Context (5ch):  {context_tokens:>10,} tokens ({context_tokens/input_tokens*100:.0f}%)")
print(f"    - User question:  {q_tokens:>10} tokens")
print(f"    ─────────────────────────────────────")
print(f"    总输入:           {input_tokens:>10,} tokens")
print(f"    输出 (max):       {output_tokens:>10} tokens")
print(f"    每次提问 LLM 调用: {input_tokens + output_tokens:>10,} tokens")

print(f"\n  费用明细（{model}）:")
print(f"    输入: RMB {pricing['input']:>6.2f} / 1M tokens → RMB {input_cost:.6f}")
print(f"    输出: RMB {pricing['output']:>6.2f} / 1M tokens → RMB {output_cost:.6f}")
print(f"    单次提问费用: RMB {total_cost:.6f}")

# ============================================================
# 5. 全链路汇总
# ============================================================
print(f"\n{'═'*70}")
print(f"  全链路费用汇总")
print(f"{'═'*70}")

# 首次上传费用
first_time = embed_cost + 0  # embedding 已计
print(f"  首次上传（{n} chunks）:")
print(f"    Embedding:      RMB {embed_cost:.6f}")

# 每次提问
print(f"\n  每次提问:")
print(f"    LLM 调用:       RMB {total_cost:.6f}")
print(f"    Embedding(问题): RMB {embed_cost / n / 10:.6f}（≈ 1/10 chunk 嵌入费）")

# 批量估算
print(f"\n  批量估算（假设 k=1,000 次提问/月）:")
monthly_llm = total_cost * 1000
print(f"    LLM:            RMB {monthly_llm:.2f}")
print(f"    文档 Embedding:  RMB {embed_cost:.2f}（一次性）")

model_name = "DeepSeek" if "deepseek" in model else "GLM"
print(f"\n  >>> 若换模型 <<<")
for alt_model, alt_price in [("deepseek-chat", PRICING["deepseek-chat"]),
                              ("deepseek-reasoner", PRICING["deepseek-reasoner"]),
                              ("glm-4-plus", PRICING["glm-4-plus"])]:
    alt_input = input_tokens / 1_000_000 * alt_price["input"]
    alt_output = output_tokens / 1_000_000 * alt_price["output"]
    alt_total = alt_input + alt_output
    print(f"    {alt_model:25s} RMB {alt_total:.6f}/次  RMB {alt_total*1000:.2f}/月")

print(f"\n{'═'*70}")
print(f"  优化建议")
print(f"{'═'*70}")

# 分析 context 占比
context_pct = context_tokens / input_tokens * 100
print(f"  - Context 占 LLM 输入的 {context_pct:.0f}%，是最大开销来源")
print(f"  - 当前 {n_results} chunks × ~{settings.chunk_size} tokens = ~{context_tokens} tokens")
print(f"  - 降低 n_results 或 chunk_size 可线性减少 LLM 输入开销")
print(f"  - 引入 Rerank 可在保持质量的同时用更少 chunk")
