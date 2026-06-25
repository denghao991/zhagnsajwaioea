"""Embedding model evaluation: real Recall@k, MRR, latency, cost.

Usage:
    # 1) 查看所有 chunk（方便填写 qa_pairs.yaml）
    python scripts/embedding_eval.py --list-chunks

    # 2) 运行完整评估
    python scripts/embedding_eval.py

    # 3) 指定自定义配置
    python scripts/embedding_eval.py --config my_config.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# Windows GBK 终端下保证中文能正常打印
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from src.tiny_rag.ingestion.chunker import MarkdownChunker
from src.tiny_rag.ingestion.tokenizer import count_tokens, encode, decode
from src.tiny_rag.generation.llm import LLMClient
from src.tiny_rag.config import settings as rag_settings


# ── 配置加载 ──────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 文档加载 & 分块 ──────────────────────────────────────

def load_chunks(doc_dir: str) -> list[dict]:
    """Load all markdown files, chunk with MarkdownChunker, return list."""
    chunker = MarkdownChunker()
    chunks: list[dict] = []

    md_files = sorted(Path(doc_dir).glob("*.md"))
    if not md_files:
        print(f"  ERROR: no .md files found in {doc_dir}")
        return chunks

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


def list_chunks(chunks: list[dict]) -> None:
    """Print all chunks with their IDs for QA pair authoring."""
    max_id_len = max(len(c["chunk_id"]) for c in chunks) if chunks else 10
    sep = "-" * 80
    print(f"\n{'Chunk ID':<{max_id_len + 2}} {'Tokens':>6}  {'Preview'}")
    print(sep)
    for c in chunks:
        preview = c["text"][:60].replace("\n", " ")
        print(f"{c['chunk_id']:<{max_id_len + 2}} {c['tokens']:>6}  {preview}")
    print(sep)
    print(f"Total: {len(chunks)} chunks\n")


# ── QA Pairs 加载 ────────────────────────────────────────

def load_qa_pairs(path: str, chunks: list[dict]) -> list[dict]:
    """Load QA pairs, validate expected chunks, warn about missing ones."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    pairs = raw.get("qa_pairs", [])
    if not pairs:
        print("  WARNING: qa_pairs is empty — no questions to evaluate.")
        return []

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
            print(f"  WARNING: question \"{q[:40]}...\" references unknown chunks: {missing}")
        if not valid_expected:
            print(f"  SKIP: question \"{q[:40]}...\" has no valid expected chunks")
            continue
        validated.append({"question": q, "expected_chunk_ids": valid_expected})

    print(f"  Loaded {len(validated)} QA pairs ({len(pairs) - len(validated)} skipped/invalid)")
    return validated


# ── Embedding（带 token 截断）────────────────────────────

def embed_batch(
    client: OpenAI, model: str, texts: list[str],
    dimensions: int | None, max_input_tokens: int | None,
    model_name: str,
) -> tuple[list[list[float]], int]:
    """Embed a batch of texts, truncating if max_input_tokens is set.

    Returns (vectors, total_truncated_chars).
    """
    truncated_chars = 0
    processed_texts: list[str] = []
    for t in texts:
        if max_input_tokens is not None:
            tokens = encode(t)
            if len(tokens) > max_input_tokens:
                truncated = decode(tokens[:max_input_tokens])
                truncated_chars += len(t) - len(truncated)
                processed_texts.append(truncated)
            else:
                processed_texts.append(t)
        else:
            processed_texts.append(t)

    kwargs: dict = {"model": model, "input": processed_texts}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    try:
        resp = client.embeddings.create(**kwargs)
    except Exception as e:
        if dimensions is not None and "dimensions" in str(e).lower():
            print(f"    [WARN] {model_name}: dimensions param not supported, retrying without it.")
            del kwargs["dimensions"]
            resp = client.embeddings.create(**kwargs)
        else:
            raise

    sorted_data = sorted(resp.data, key=lambda x: x.index)
    actual_dim = len(sorted_data[0].embedding)
    if dimensions is not None and actual_dim != dimensions:
        print(f"    [WARN] {model_name}: returned {actual_dim} dims, expected {dimensions}")

    return [d.embedding for d in sorted_data], truncated_chars


# ── 检索 ──────────────────────────────────────────────────

def search(query_vec: list[float], chunk_vectors: np.ndarray,
           chunk_ids: list[str], top_k: int = 20) -> list[tuple[str, float]]:
    """Cosine similarity search, return [(chunk_id, score), ...] sorted desc."""
    q = np.array(query_vec, dtype=np.float32)
    norms = np.linalg.norm(chunk_vectors, axis=1)
    similarities = (chunk_vectors @ q) / (norms * np.linalg.norm(q) + 1e-10)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(chunk_ids[i], float(similarities[i])) for i in top_indices]


# ── 单模型评估 ────────────────────────────────────────────

def evaluate_model(
    model_cfg: dict, chunks: list[dict], qa_pairs: list[dict],
) -> dict | None:
    """Evaluate one embedding model on the QA pairs.

    Returns result dict or None if evaluation fails.
    """
    name = model_cfg["name"]
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"{'='*60}")

    api_key = os.getenv(model_cfg["api_key_env"])
    if not api_key:
        print(f"  SKIP: {model_cfg['api_key_env']} not set in .env")
        return None

    client = OpenAI(api_key=api_key, base_url=model_cfg["base_url"])
    model_name = model_cfg["model"]
    dimensions = model_cfg.get("dimensions")
    batch_size = model_cfg.get("batch_size", 16)
    max_input_tokens = model_cfg.get("max_input_tokens")

    # ── Step 1: Embed all chunks ──
    print(f"  Embedding {len(chunks)} chunks (batch_size={batch_size})...")
    all_vectors: list[list[float]] = []
    batch_latencies: list[float] = []
    total_input_tokens = 0
    total_truncated_chars = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [d["text"] for d in batch]
        total_input_tokens += sum(count_tokens(t) for t in texts)

        start = time.perf_counter()
        vectors, tc = embed_batch(
            client, model_name, texts, dimensions, max_input_tokens, name,
        )
        elapsed = time.perf_counter() - start

        batch_latencies.append(elapsed)
        all_vectors.extend(vectors)
        total_truncated_chars += tc
        print(f"    [{i + len(batch):>4}/{len(chunks)}] batch {elapsed:.2f}s")

    chunk_vector_matrix = np.array(all_vectors, dtype=np.float32)
    chunk_ids = [c["chunk_id"] for c in chunks]

    if total_truncated_chars > 0:
        print(f"  [INFO] Truncated {total_truncated_chars} chars total across chunks")

    # ── Step 2: Evaluate each QA pair ──
    print(f"  Evaluating {len(qa_pairs)} questions...")
    ks = [1, 3, 5, 10, 20]
    recall_buffer: dict[int, list[float]] = {k: [] for k in ks}
    reciprocal_ranks: list[float] = []
    per_question: list[dict] = []

    # LLM rewrite client（与 RAG 生产环境使用相同模型，5s 超时避免卡住）
    rewrite_llm = LLMClient(
        base_url=rag_settings.llm_base_url,
        api_key=rag_settings.llm_api_key,
        model=rag_settings.llm_model,
    )
    # 给底层 HTTP 客户端设短超时，API 不可用时快速降级
    rewrite_llm._client.timeout = 5

    for pq in qa_pairs:
        q = pq["question"]
        expected = set(pq["expected_chunk_ids"])

        # 查询改写：将口语/缩写展开为文档术语
        rewritten = rewrite_llm.rewrite(q)
        if rewritten != q:
            print(f"    rewrite: \"{q}\" → \"{rewritten}\"")

        # Embed 改写后的问题
        q_vecs, _ = embed_batch(
            client, model_name, [rewritten], dimensions, max_input_tokens, name,
        )
        q_vec = q_vecs[0]

        # Search
        results = search(q_vec, chunk_vector_matrix, chunk_ids, top_k=max(ks))

        top_k_ids: dict[int, set[str]] = {}
        for k in ks:
            top_k_ids[k] = {rid for rid, _ in results[:k]}

        first_rank: int | None = None
        for rank, (rid, _) in enumerate(results, start=1):
            if rid in expected:
                first_rank = rank
                break

        rr = 1.0 / first_rank if first_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        for k in ks:
            hit = 1.0 if expected & top_k_ids[k] else 0.0
            recall_buffer[k].append(hit)

        per_question.append({
            "question": q,
            "rewritten": rewritten,
            "expected": list(expected),
            "first_rank": first_rank,
            "reciprocal_rank": rr,
            "top_5": [{"chunk_id": rid, "score": round(s, 4)} for rid, s in results[:5]],
        })

    recalls = {k: float(np.mean(v)) for k, v in recall_buffer.items()}
    mrr = float(np.mean(reciprocal_ranks))

    # ── Step 3: Latency stats ──
    lat_arr = np.array(batch_latencies)
    lat_p50 = float(np.percentile(lat_arr, 50))
    lat_p95 = float(np.percentile(lat_arr, 95))
    lat_p99 = float(np.percentile(lat_arr, 99))
    total_time = float(lat_arr.sum())
    per_req_p50 = lat_p50 / batch_size
    per_req_p95 = lat_p95 / batch_size

    # ── Step 4: Cost estimate ──
    price = model_cfg.get("price_per_million_tokens")
    cost_usd: float | None = None
    if price is not None:
        cost_usd = total_input_tokens / 1_000_000 * price
    cost_str = f"${cost_usd:.4f}" if cost_usd is not None else "N/A"

    # ── Print ──
    print(f"\n  --- Results: {name} ---")
    print(f"  Recall@1:  {recalls[1]:.4f}")
    print(f"  Recall@3:  {recalls[3]:.4f}")
    print(f"  Recall@5:  {recalls[5]:.4f}")
    print(f"  Recall@10: {recalls[10]:.4f}")
    print(f"  Recall@20: {recalls[20]:.4f}")
    print(f"  MRR:       {mrr:.4f}")
    print(f"\n  Latency (batch={batch_size}):")
    print(f"    P50: {lat_p50:.2f}s  (per-req ~{per_req_p50:.3f}s)")
    print(f"    P95: {lat_p95:.2f}s  (per-req ~{per_req_p95:.3f}s)")
    print(f"    P99: {lat_p99:.2f}s")
    print(f"    Total: {total_time:.2f}s")
    print(f"\n  Cost: {cost_str}")
    if cost_usd is not None:
        print(f"    Input tokens: {total_input_tokens}")
        print(f"    Rate: ${price}/M tokens")

    result = {
        "model": name,
        "dimensions": dimensions or "default",
        "max_input_tokens": max_input_tokens,
        "num_chunks": len(chunks),
        "num_questions": len(qa_pairs),
        "recall": {str(k): round(v, 4) for k, v in recalls.items()},
        "mrr": round(mrr, 4),
        "truncated_chars": total_truncated_chars,
        "latency": {
            "p50": round(lat_p50, 2),
            "p95": round(lat_p95, 2),
            "p99": round(lat_p99, 2),
            "per_req_p50": round(per_req_p50, 4),
            "per_req_p95": round(per_req_p95, 4),
            "total_seconds": round(total_time, 2),
        },
        "cost": {
            "total_usd": round(cost_usd, 6) if cost_usd is not None else None,
            "price_per_million_tokens": price,
            "input_tokens": total_input_tokens,
        },
        "per_question": per_question,
    }

    safe_name = name.replace("/", "_").replace(" ", "_")
    out_path = _project_root / "data" / "eval" / "results" / f"{safe_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Detail saved to {out_path}")

    return result


# ── 汇总 ──────────────────────────────────────────────────

def print_header():
    print(f"\n{'='*110}")
    h = (
        f"{'Model':<38} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7}"
        f" {'MRR':>7} {'P50(s)':>7} {'P95(s)':>7} {'Cost($)':>10}"
    )
    print(h)
    print(f"{'-'*110}")


def print_row(name: str, recalls: dict, mrr: float, lat_p50: float,
              lat_p95: float, cost: str):
    print(
        f"{name:<38}"
        f" {recalls[1]:>6.3f}"
        f" {recalls[3]:>6.3f}"
        f" {recalls[5]:>6.3f}"
        f" {recalls[10]:>6.3f}"
        f" {recalls[20]:>6.3f}"
        f" {mrr:>6.3f}"
        f" {lat_p50:>6.2f}"
        f" {lat_p95:>6.2f}"
        f" {cost:>10}"
    )


# ── CLI ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embedding model evaluation")
    parser.add_argument(
        "--config",
        default=str(_project_root / "scripts" / "embedding_eval_config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--list-chunks",
        action="store_true",
        help="Print all chunks and exit (use this to fill qa_pairs.yaml)",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    doc_dir = str((_project_root / config["doc_dir"]).resolve())

    # ── Load chunks ──
    print(f"Loading docs from {doc_dir} ...")
    chunks = load_chunks(doc_dir)
    if not chunks:
        print("ERROR: no chunks loaded, exit.")
        sys.exit(1)

    num_docs = len({c["doc_id"] for c in chunks})
    print(f"  {len(chunks)} chunks from {num_docs} files")
    print(f"  total tokens: {sum(c['tokens'] for c in chunks)}")

    # ── --list-chunks mode ──
    if args.list_chunks:
        list_chunks(chunks)
        return

    # ── Load QA pairs ──
    qa_path = str((_project_root / config["qa_pairs_path"]).resolve())
    print(f"Loading QA pairs from {qa_path} ...")
    qa_pairs = load_qa_pairs(qa_path, chunks)

    if not qa_pairs:
        print(
            "ERROR: no valid QA pairs. "
            "Fill data/eval/qa_pairs.yaml first, or run with --list-chunks to see available chunks."
        )
        sys.exit(1)

    # ── Evaluate each model ──
    models_cfg = config.get("models", [])
    if not models_cfg:
        print("ERROR: no models defined in config")
        sys.exit(1)

    all_results: dict[str, dict] = {}

    for mc in models_cfg:
        try:
            result = evaluate_model(mc, chunks, qa_pairs)
        except Exception as e:
            print(f"\n  [ERROR] {mc['name']} failed: {e}")
            print(f"  SKIP this model and continue.")
            continue
        if result is not None:
            all_results[mc["name"]] = result

    # ── Summary table ──
    if all_results:
        print(f"\n\n{'='*110}")
        print("  SUMMARY")
        print_header()
        for name, r in all_results.items():
            lat = r["latency"]
            cost_val = r["cost"]["total_usd"]
            cost_disp = f"${cost_val:.4f}" if cost_val is not None else "N/A"
            print_row(
                name,
                {int(k): v for k, v in r["recall"].items()},
                r["mrr"],
                lat["p50"],
                lat["p95"],
                cost_disp,
            )

        # ── Best pick suggestion ──
        print(f"\n  {'─'*110}")
        print(f"  Shortlist (weighted: R@5 + MRR - 0.02*P50):")
        scored = []
        for name, r in all_results.items():
            score = r["recall"]["5"] + r["mrr"] - 0.02 * r["latency"]["p50"]
            scored.append((name, score, r))
        scored.sort(key=lambda x: x[1], reverse=True)
        for name, score, r in scored:
            print(f"    {name:<38} score={score:.4f}  "
                  f"(R@5={r['recall']['5']:.3f}, MRR={r['mrr']:.3f}, P50={r['latency']['p50']:.2f}s)")
    else:
        print("\nNo models evaluated successfully.")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
