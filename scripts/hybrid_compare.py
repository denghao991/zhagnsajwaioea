"""对比测试：纯向量检索 vs 混合检索（向量 + BM25 + RRF）。

用法：PYTHONPATH=. python scripts/hybrid_compare.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tiny_rag.embedding.client import EmbeddingClient
from src.tiny_rag.storage.vector_store import VectorStore
from src.tiny_rag.retrieval.bm25 import BM25Retriever
from src.tiny_rag.retrieval.hybrid import rrf_merge
from src.tiny_rag.ingestion.chunker import ChunkResult
from src.tiny_rag.config import settings

# ── 文档条目（用户提供）──
DOC_ITEMS: dict[int, str] = {
    1: "CSS内存不足检查项位于xx平台,核心逻辑是先查询CSS资源,然后调取CES监控,返回结果字段CSS实例ID,以及近七天的内存使用率",
    2: "CSS内存不足检查项判断逻辑标准是近7天用户内存使用率高于80%",
    3: "可用性检查权限不足需要去先在可用性检查页面编辑自己为技术专家,如果没有该用户的权限,需要联系OA oncall",
    4: "容量优化执行一直排队中是由于当前资源有限,请耐心等待",
    5: "容量优化CCE的执行逻辑是调用AOM接口,获取指定规则的监控数据,组装规则不由OA控制,如果有疑问可以联系AOM oncall",
    6: "容量优化CCE推荐使用CPU+内存+IO这三种方式去判断资源风险",
}

# ── 测试问题 → 期望命中的条目编号 ──
TEST_CASES: list[dict] = [
    {"question": "可用性检查CSS的检查逻辑是啥", "expected": {1, 2}},
    {"question": "容量优化点了没反应",          "expected": {4}},
    {"question": "容量的CCE一直排队",           "expected": {4}},
    {"question": "CCE容量结果不对啥原因",        "expected": {5}},
]


def main() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # ── 初始化 ──
        print("初始化 Embedding 客户端...")
        embedder = EmbeddingClient(
            base_url=settings.dashscope_base_url,
            api_key=settings.dashscope_api_key,
            model=settings.embedding_model,
        )
        vector_store = VectorStore(persist_dir=tmpdir)
        bm25_retriever = BM25Retriever()

        # ── 文档入库 ──
        texts = [DOC_ITEMS[i] for i in sorted(DOC_ITEMS)]
        print(f"\n=== 文档入库（{len(texts)} 条）===")
        for idx, text in DOC_ITEMS.items():
            print(f"  [{idx}] {text[:50]}...")

        print("  生成 Embedding...")
        embeddings = embedder.embed(texts)

        chunk_results = [ChunkResult(text=t) for t in texts]
        vector_store.add_document(
            doc_id="doc_test", filename="test.md",
            chunks=chunk_results, embeddings=embeddings,
        )
        bm25_retriever.add_document(doc_id="doc_test", filename="test.md", chunks=texts)

        # ── 逐条测试 ──
        vector_ok, hybrid_ok = 0, 0

        for tc in TEST_CASES:
            q = tc["question"]
            expected = tc["expected"]

            print(f"\n{'='*60}")
            print(f"问题: {q}")
            print(f"期望命中条目: {expected}")

            # 向量检索
            q_emb = embedder.embed([q])[0]
            vec_results = vector_store.search(q_emb, n_results=5)
            vec_hit = {n for n, t in DOC_ITEMS.items() if t in {r["text"] for r in vec_results}}
            vec_pass = bool(expected & vec_hit)
            if vec_pass:
                vector_ok += 1

            # 混合检索
            bm25_results = bm25_retriever.search(q, n_results=10)
            hybrid_results = rrf_merge(vec_results, bm25_results, n_results=5)
            hybrid_hit = {n for n, t in DOC_ITEMS.items() if t in {r["text"] for r in hybrid_results}}
            hybrid_pass = bool(expected & hybrid_hit)
            if hybrid_pass:
                hybrid_ok += 1

            # 打印对比
            status = "OK" if vec_pass else "MISS"
            print(f"  [{status}] 向量 TOP5: [{', '.join(f'{i}({_dist(vec_hit, vec_results, i)})' for i in sorted(vec_hit)) or '无'}]")

            status = "OK" if hybrid_pass else "MISS"
            print(f"  [{status}] 混合 TOP5: [{', '.join(str(i) for i in sorted(hybrid_hit)) or '无'}]")

            # 打印 BM25 原始结果
            print(f"  BM25 TOP5:")
            for rank, r in enumerate(bm25_results[:5], 1):
                item_idx = next((i for i, t in DOC_ITEMS.items() if t == r["text"]), None)
                print(f"    #{rank} item[{item_idx}] score={r['score']:.2f}  {r['text'][:45]}...")

            # 打印混合结果的来源分布
            print(f"  RRF 合并结果来源:")
            for rank, r in enumerate(hybrid_results, 1):
                item_idx = next((i for i, t in DOC_ITEMS.items() if t == r["text"]), None)
                in_vec = item_idx in vec_hit if item_idx else False
                in_bm25 = item_idx in {n for n, t in DOC_ITEMS.items() if t in {br["text"] for br in bm25_results}} if item_idx else False
                src = []
                if in_vec: src.append("向量")
                if in_bm25: src.append("BM25")
                print(f"    #{rank} item[{item_idx}] 来自={'+'.join(src) if src else '?'}  {r['text'][:45]}...")

        # ── 汇总 ──
        total = len(TEST_CASES)
        print(f"\n{'='*60}")
        print(f"          总问题数: {total}")
        print(f"  向量检索命中率: {vector_ok}/{total} ({vector_ok/total*100:.0f}%)")
        print(f"  混合检索命中率: {hybrid_ok}/{total} ({hybrid_ok/total*100:.0f}%)")

        if hybrid_ok > vector_ok:
            print(f"\n  BM25 额外挽回了 {hybrid_ok - vector_ok} 个问题 ✓")
        elif hybrid_ok == vector_ok:
            print(f"\n  两者持平，但混合检索排序可能更优")
        else:
            print(f"\n  向量检索反而更好，可能需要调 BM25 参数")


def _dist(hits: set, results: list[dict], item_idx: int) -> str:
    """Format distance for display."""
    for r in results:
        if r.get("text") == DOC_ITEMS.get(item_idx):
            return f"{r.get('distance', 0):.3f}"
    return ""


if __name__ == "__main__":
    main()
