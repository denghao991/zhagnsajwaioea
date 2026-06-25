"""RRF weight evaluation — compare (alpha, beta) on real QA pairs.

用法：
    python scripts/rrf_eval.py

对每组 (alpha, beta) 组合在同一组文档 + QA pairs 上跑 RRF 合并，
输出 Recall@K / MRR / 来源分布对比表。

Embedding 只跑一次，各组之间共享向量矩阵和 BM25 索引。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.tiny_rag.ingestion.chunker import MarkdownChunker
from src.tiny_rag.ingestion.tokenizer import count_tokens
from src.tiny_rag.retrieval.hybrid import rrf_merge
from src.tiny_rag.retrieval.bm25 import BM25Retriever
from src.tiny_rag.config import settings
from src.tiny_rag.embedding.client import EmbeddingClient
from src.tiny_rag.generation.llm import LLMClient

# ── 候选 N 组合（固定 alpha=7, beta=3）──
N_GRID: list[dict] = [
    {"vector_n": 5,  "bm25_n": 3,  "label": "V5 +B3"},
    {"vector_n": 10, "bm25_n": 3,  "label": "V10+B3"},
    {"vector_n": 10, "bm25_n": 5,  "label": "V10+B5"},
    {"vector_n": 15, "bm25_n": 5,  "label": "V15+B5"},
    {"vector_n": 15, "bm25_n": 10, "label": "V15+B10"},
    {"vector_n": 20, "bm25_n": 10, "label": "V20+B10"},
    {"vector_n": 20, "bm25_n": 15, "label": "V20+B15"},
    {"vector_n": 30, "bm25_n": 10, "label": "V30+B10"},
    {"vector_n": 30, "bm25_n": 15, "label": "V30+B15"},
]

ALPHA: float = 7.0
BETA: float = 3.0

KS = [1, 3, 5, 10]


# ── 文档加载 & 分块 ──────────────────────────────────────

def load_chunks(doc_dir: str) -> list[dict]:
    """Load all markdown files, chunk with MarkdownChunker, return list."""
    chunker = MarkdownChunker()
    chunks: list[dict] = []
    md_files = sorted(Path(doc_dir).glob("*.md"))
    for fpath in md_files:
        text = fpath.read_text(encoding="utf-8")
        results = chunker.chunk_text(text)
        for i, c in enumerate(results):
            chunks.append({
                "doc_id": fpath.stem,
                "chunk_id": f"{fpath.stem}#{i}",
                "text": c.text,
                "tokens": c.token_count,
            })
    return chunks


# ── QA Pairs 加载 ────────────────────────────────────────

def load_qa_pairs(path: str, chunks: list[dict]) -> list[dict]:
    """Load QA pairs, validate expected chunks against actual chunks."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    pairs = raw.get("qa_pairs", [])
    valid_ids = {c["chunk_id"] for c in chunks}
    validated: list[dict] = []
    for p in pairs:
        q = p.get("question", "").strip()
        expected = p.get("expected_chunk_ids", [])
        if not q:
            continue
        valid_expected = [eid for eid in expected if eid in valid_ids]
        missing = set(expected) - valid_ids
        if missing:
            print(f"  WARNING: \"{q[:40]}...\" references unknown chunks: {missing}")
        if not valid_expected:
            continue
        validated.append({"question": q, "expected_chunk_ids": valid_expected})
    return validated


# ── 向量搜索（余弦相似度）──────────────────────────────────

def search(
    query_vec: list[float],
    chunk_vectors: np.ndarray,
    chunk_ids: list[str],
    text_map: dict[str, str],
    top_k: int = 20,
) -> list[dict]:
    """Cosine similarity search, return list of dicts with 'text' key."""
    q = np.array(query_vec, dtype=np.float32)
    norms = np.linalg.norm(chunk_vectors, axis=1)
    similarities = (chunk_vectors @ q) / (norms * np.linalg.norm(q) + 1e-10)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [
        {
            "text": text_map[chunk_ids[i]],
            "chunk_id": chunk_ids[i],
            "score": float(similarities[i]),
        }
        for i in top_indices
    ]


# ── 单 (VECTOR_N, BM25_N) 组合评估 ─────────────────────

def evaluate_n(
    qa_pairs: list[dict],
    chunk_vector_matrix: np.ndarray,
    chunk_ids: list[str],
    text_map: dict[str, str],
    reverse_text_map: dict[str, str],
    bm25: BM25Retriever,
    embedder: EmbeddingClient,
    vector_n: int,
    bm25_n: int,
) -> dict:
    """Run all QA pairs with given VECTOR_N & BM25_N, return aggregated metrics."""
    all_recalls: dict[int, list[float]] = {k: [] for k in KS}
    reciprocal_ranks: list[float] = []
    total_source_dist: dict[str, int] = {"vector": 0, "bm25": 0, "both": 0}
    per_question: list[dict] = []

    for pq in qa_pairs:
        q = pq["rewritten"]  # 用改写后的问题
        orig_q = pq["question"]
        expected_set = set(pq["expected_chunk_ids"])

        # 向量检索
        q_vecs = embedder.embed([q])
        vec_results = search(
            q_vecs[0], chunk_vector_matrix, chunk_ids, text_map,
            top_k=vector_n,
        )
        vec_texts = {r["text"] for r in vec_results}

        # BM25 检索
        bm25_results = bm25.search(q, n_results=bm25_n)
        bm25_texts = {r["text"] for r in bm25_results}

        # RRF 合并（固定 alpha=7, beta=3）
        merged = rrf_merge(vec_results, bm25_results, n_results=10,
                           alpha=ALPHA, beta=BETA)

        # 来源分布
        for r in merged:
            in_v = r["text"] in vec_texts
            in_b = r["text"] in bm25_texts
            if in_v and in_b:
                total_source_dist["both"] += 1
            elif in_v:
                total_source_dist["vector"] += 1
            elif in_b:
                total_source_dist["bm25"] += 1

        merged_ids = {reverse_text_map.get(r["text"], "") for r in merged}

        # 找到期望 chunk 的最早排名
        first_rank: int | None = None
        for rank, r in enumerate(merged, start=1):
            if reverse_text_map.get(r["text"]) in expected_set:
                first_rank = rank
                break
        rr = 1.0 / first_rank if first_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        for k in KS:
            top_k_ids = {reverse_text_map.get(r["text"]) for r in merged[:k]}
            hit = 1.0 if expected_set & top_k_ids else 0.0
            all_recalls[k].append(hit)

        per_question.append({
            "question": orig_q,
            "expected": list(expected_set),
            "first_rank": first_rank,
            "reciprocal_rank": rr,
            "top_5": [{"chunk_id": reverse_text_map.get(r["text"], "?"),
                       "score": r.get("score", 0.0)}
                      for r in merged[:5]],
        })

    recalls = {k: float(np.mean(v)) for k, v in all_recalls.items()}
    mrr = float(np.mean(reciprocal_ranks))
    n_questions = len(qa_pairs)

    return {
        "vector_n": vector_n,
        "bm25_n": bm25_n,
        "recall": {str(k): round(recalls[k], 4) for k in KS},
        "mrr": round(mrr, 4),
        "source_distribution": total_source_dist,
        "n_questions": n_questions,
        "per_question": per_question,
    }


# ── 主流程 ────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RRF weight evaluation")
    parser.add_argument("--doc-dir", default="data/eval/md",
                        help="文档目录，默认 data/eval/md")
    parser.add_argument("--qa-pairs", default="data/eval/qa_pairs.yaml",
                        help="QA pairs 路径，默认 data/eval/qa_pairs.yaml")
    parser.add_argument("-o", "--output", type=str,
                        help="保存完整结果到 JSON 文件")
    args = parser.parse_args()

    doc_dir = str((_project_root / args.doc_dir).resolve())
    qa_path = str((_project_root / args.qa_pairs).resolve())

    # ── 1. 加载文档 & 分块 ──
    print(f"加载文档: {doc_dir}")
    chunks = load_chunks(doc_dir)
    if not chunks:
        print("ERROR: 没有加载到任何文档")
        sys.exit(1)
    print(f"  {len(chunks)} chunks from {len({c['doc_id'] for c in chunks})} files")

    # ── 2. 加载 QA pairs ──
    qa_pairs = load_qa_pairs(qa_path, chunks)
    if len(qa_pairs) < 2:
        print(f"ERROR: 有效 QA pairs 不足（{len(qa_pairs)}），至少需要 2 条")
        sys.exit(1)
    print(f"  {len(qa_pairs)} QA pairs\n")

    # ── 3. 查询改写（一次性）──
    print("初始化 LLM 改写客户端...")
    rewrite_llm = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    rewrite_llm._client.timeout = 5
    print("  改写 QA pairs...")
    rewritten_count = 0
    for pq in qa_pairs:
        original = pq["question"]
        rewritten = rewrite_llm.rewrite(original)
        pq["rewritten"] = rewritten
        if rewritten != original:
            rewritten_count += 1
            print(f"    \"{original[:40]}...\" → \"{rewritten[:40]}...\"")
    print(f"  {rewritten_count}/{len(qa_pairs)} 条被改写\n")

    # ── 4. 向量化（一次性）──
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    text_map = dict(zip(chunk_ids, texts))
    reverse_text_map = {t: cid for cid, t in text_map.items()}

    print("初始化 Embedding 客户端...")
    embedder = EmbeddingClient(
        base_url=settings.dashscope_base_url,
        api_key=settings.dashscope_api_key,
        model=settings.embedding_model,
    )
    print(f"  Embedding {len(chunks)} chunks...")
    batch_size = 20
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        all_vectors.extend(embedder.embed(batch))
        print(f"    [{i + len(batch):>4}/{len(chunks)}] embedded")
    chunk_vector_matrix = np.array(all_vectors, dtype=np.float32)

    # ── 5. BM25 索引（一次性）──
    print("  构建 BM25 索引...")
    bm25 = BM25Retriever()
    bm25.add_document(doc_id="eval", filename="eval", chunks=texts)

    # ── 6. 对每组 N 跑评估 ──
    results: list[dict] = []
    for nc in N_GRID:
        vn, bn, label = nc["vector_n"], nc["bm25_n"], nc["label"]
        print(f"\n  ── VECTOR_N={vn}, BM25_N={bn} ({label}) ──")
        r = evaluate_n(
            qa_pairs, chunk_vector_matrix, chunk_ids, text_map,
            reverse_text_map, bm25, embedder, vn, bn,
        )
        results.append(r)
        # 打印本组合摘要
        print(f"    Recall@1: {r['recall']['1']:.3f}  "
              f"Recall@5: {r['recall']['5']:.3f}  "
              f"MRR: {r['mrr']:.3f}")
        src = r["source_distribution"]
        print(f"    来源: 向量={src['vector']}  BM25={src['bm25']}  共同={src['both']}")

    # ── 6. 汇总对比表 ──
    print(f"\n{'='*100}")
    print(f"  召回深度（N）评估汇总  |  {len(qa_pairs)} QA pairs, {len(chunks)} chunks"
          f"  |  alpha={ALPHA}, beta={BETA}")
    print(f"{'='*100}")
    header = (
        f"  {'组合':<12} {'V_N':>5} {'B_N':>5}"
        f"  {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6}"
        f"  {'MRR':>6}  {'来源分布'}"
    )
    print(header)
    print(f"  {'-'*12} {'-'*5} {'-'*5}  {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*6}  {'-'*20}")
    for r in results:
        src = r["source_distribution"]
        src_str = f"V:{src['vector']} B:{src['bm25']} V+B:{src['both']}"
        label = next(nc["label"] for nc in N_GRID
                     if nc["vector_n"] == r["vector_n"]
                     and nc["bm25_n"] == r["bm25_n"])
        print(
            f"  {label:<12} {r['vector_n']:>5} {r['bm25_n']:>5}"
            f"  {r['recall']['1']:>6.3f} {r['recall']['3']:>6.3f}"
            f"  {r['recall']['5']:>6.3f} {r['recall']['10']:>6.3f}"
            f"  {r['mrr']:>6.3f}  {src_str}"
        )

    # ── 7. 推荐 ──
    print(f"\n  推荐分析:")
    scored = []
    for r in results:
        # 综合得分：R@5 + MRR
        score = r["recall"]["5"] + r["mrr"]
        scored.append((r["vector_n"], r["bm25_n"], score))
    scored.sort(key=lambda x: x[2], reverse=True)
    best_vn, best_bn, best_score = scored[0]
    best_label = next(nc["label"] for nc in N_GRID
                      if nc["vector_n"] == best_vn and nc["bm25_n"] == best_bn)
    print(f"    综合得分（R@5 + MRR）:")
    for vn, bn, s in scored:
        label = next(nc["label"] for nc in N_GRID
                     if nc["vector_n"] == vn and nc["bm25_n"] == bn)
        print(f"      V={vn} B={bn} ({label}) → {s:.4f}")
    print(f"    推荐: VECTOR_N={best_vn}, BM25_N={best_bn} ({best_label})")

    # ── 8. 保存 JSON ──
    if args.output:
        out_path = (_project_root / args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "n_qa_pairs": len(qa_pairs),
                "n_chunks": len(chunks),
                "doc_files": sorted({c["doc_id"] for c in chunks}),
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存到: {out_path}")

    print()


if __name__ == "__main__":
    main()
