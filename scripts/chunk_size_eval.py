"""Chunk size 评估 — 统计不同 chunk_size 下分块的质量指标。

用法：
    # 使用默认文档目录 (data/eval/md/) 和默认 chunk_size 列表
    python scripts/chunk_size_eval.py

    # 指定文档目录
    python scripts/chunk_size_eval.py --doc-dir data/eval/md

    # 指定 chunk_size 列表
    python scripts/chunk_size_eval.py --chunk-sizes 256 384 512 768

    # 保存结果到 JSON
    python scripts/chunk_size_eval.py -o data/eval/chunk_size_results.json
"""

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# Windows GBK 终端下保证中文能正常打印
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.tiny_rag.ingestion.chunker import MarkdownChunker
from src.tiny_rag.ingestion.tokenizer import count_tokens

import numpy as np

SHORT_THRESHOLD = 50  # tokens 低于此值视为碎片


def load_documents(doc_dir: str) -> list[tuple[str, str]]:
    """加载目录下所有 .md 文件。返回 [(文件名, 全文)]。"""
    path = Path(doc_dir)
    files = sorted(path.glob("*.md"))
    if not files:
        print(f"  ERROR: 在 {doc_dir} 中没有找到 .md 文件")
        return []
    docs = []
    for f in files:
        docs.append((f.name, f.read_text(encoding="utf-8")))
        print(f"  + {f.name}")
    return docs


def evaluate_chunk_size(docs: list[tuple[str, str]],
                        chunk_size: int,
                        overlap: int = 0) -> dict:
    """对一组文档用指定 chunk_size 分块，返回统计指标。"""
    chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    all_token_counts: list[int] = []
    filename_counts: dict[str, int] = {}

    for fname, text in docs:
        chunks = chunker.chunk_text(text)
        filename_counts[fname] = len(chunks)
        for c in chunks:
            all_token_counts.append(c.token_count)

    arr = np.array(all_token_counts)
    n = len(arr)

    if n == 0:
        return {
            "chunk_size": chunk_size,
            "total_chunks": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "min": 0,
            "max": 0,
            "short_pct": 0.0,
            "short_count": 0,
            "overflow_pct": 0.0,
            "overflow_count": 0,
            "fill_rate": 0.0,
            "per_file": {},
        }

    short_count = int((arr < SHORT_THRESHOLD).sum())

    # RecursiveCharacterTextSplitter 保证不超 chunk_size，但验证一下
    overflow_count = int((arr > chunk_size).sum())

    result = {
        "chunk_size": chunk_size,
        "total_chunks": n,
        "mean": float(round(arr.mean(), 1)),
        "std": float(round(arr.std(), 1)),
        "median": float(round(np.median(arr), 1)),
        "p75": float(round(np.percentile(arr, 75), 1)),
        "p90": float(round(np.percentile(arr, 90), 1)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "short_pct": round(short_count / n * 100, 1),
        "short_count": short_count,
        "overflow_pct": round(overflow_count / n * 100, 1),
        "overflow_count": overflow_count,
        "fill_rate": round(np.median(arr) / chunk_size * 100, 1),
        "per_file": {fname: count for fname, count in filename_counts.items()},
    }
    return result


def format_line(r: dict) -> str:
    """格式化为一行摘要。"""
    return (
        f"chunk_size={r['chunk_size']:<5}"
        f"  {r['total_chunks']:>4} chunks  "
        f"| 中位数 {r['median']:<6}  "
        f"| P75 {r['p75']:<6}"
        f"| P90 {r['p90']:<6}"
        f"| 均值 {r['mean']:<6} "
        f"| 碎片率 {r['short_pct']:>5.1f}% ({r['short_count']})  "
        f"| 超限 {r['overflow_pct']:>5.1f}%  "
        f"| 填充率 {r['fill_rate']:>4.1f}%"
    )


def format_detail(r: dict, docs: list[tuple[str, str]]) -> list[str]:
    """每文件的详细统计。"""
    lines = []
    for fname, _ in docs:
        cnt = r["per_file"].get(fname, 0)
        lines.append(f"      {fname:<30s}  {cnt} chunks")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Chunk size 评估工具")
    parser.add_argument("--doc-dir", default="data/eval/md",
                        help="文档目录，默认 data/eval/md")
    parser.add_argument("--chunk-sizes", nargs="+", type=int,
                        default=[128, 256, 384, 512, 768, 1024],
                        help="chunk_size 列表，默认 128 256 384 512 768 1024")
    parser.add_argument("--overlap", type=int, default=0,
                        help="chunk_overlap，默认 0")
    parser.add_argument("-o", "--output", type=str,
                        help="保存完整结果到 JSON 文件")
    args = parser.parse_args()

    doc_dir = str((_project_root / args.doc_dir).resolve())

    # ── 加载文档 ──
    print(f"加载文档: {doc_dir}")
    docs = load_documents(doc_dir)
    if not docs:
        sys.exit(1)

    # ── 跑每个 chunk_size ──
    results: list[dict] = []
    print(f"\n{'='*90}")
    print(f"  Chunk Size 评估 | 文档数: {len(docs)} | overlap: {args.overlap}")
    print(f"{'='*90}")

    for cs in args.chunk_sizes:
        r = evaluate_chunk_size(docs, cs, overlap=args.overlap)
        results.append(r)
        print(f"\n{format_line(r)}")
        detail = format_detail(r, docs)
        for d in detail:
            print(d)

    # ── 汇总决策表 ──
    print(f"\n{'='*90}")
    print(f"  汇总")
    print(f"{'='*90}")
    print(f"  {'chunk_size':>10} | {'总数':>5} | {'中位数':>7} | {'碎片率':>7} | {'填充率':>7} | {'P90':>7}")
    print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for r in results:
        if r["total_chunks"] > 0:
            print(
                f"  {r['chunk_size']:>10} |"
                f" {r['total_chunks']:>5} |"
                f" {r['median']:>7} |"
                f" {r['short_pct']:>6.1f}% |"
                f" {r['fill_rate']:>6.1f}% |"
                f" {r['p90']:>7}"
            )

    # ── 推荐 ──
    print(f"\n  推荐分析:")
    candidates = [r for r in results if r["total_chunks"] > 0
                  and r["short_pct"] <= 20
                  and r["fill_rate"] >= 30]
    if candidates:
        best = max(candidates, key=lambda x: x["fill_rate"])
        print(f"    排除碎片率 > 20% 或填充率 < 30% 的配置")
        print(f"    剩余候选: {[r['chunk_size'] for r in candidates]}")
        print(f"    推荐: chunk_size = {best['chunk_size']} (填充率 {best['fill_rate']}%)")
    else:
        print(f"    没有完全满足条件的配置，需结合文档特征人工判断")

    # ── 保存 JSON ──
    if args.output:
        out_path = (_project_root / args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "doc_dir": doc_dir,
                "doc_count": len(docs),
                "doc_files": [fname for fname, _ in docs],
                "doc_char_counts": [len(text) for _, text in docs],
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存到: {out_path}")

    print()


if __name__ == "__main__":
    main()
