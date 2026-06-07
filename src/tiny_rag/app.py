"""Flask application — RAG system web server."""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import (Flask, Response, jsonify, request, render_template,
                   stream_with_context)

from src.tiny_rag.config import (settings, VECTOR_N, BM25_N,
                                  CACHE_THRESHOLD, CACHE_MAX_ENTRIES)
from src.tiny_rag.ingestion.loader import load_bytes, load_pdf
from src.tiny_rag.ingestion.chunker import MarkdownChunker
from src.tiny_rag.ingestion.web_loader import WebLoader
from src.tiny_rag.embedding.client import EmbeddingClient
from src.tiny_rag.storage.vector_store import VectorStore
from src.tiny_rag.generation.llm import LLMClient
from src.tiny_rag.cache.semantic_cache import SemanticCache
from src.tiny_rag.retrieval.bm25 import BM25Retriever
from src.tiny_rag.retrieval.hybrid import rrf_merge
from src.tiny_rag.retrieval.reranker import RerankClient

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

app = Flask(__name__)

embedder = EmbeddingClient(
    base_url=settings.dashscope_base_url,
    api_key=settings.dashscope_api_key,
    model=settings.embedding_model,
)

vector_store = VectorStore(persist_dir=settings.chroma_persist_dir)

cache = SemanticCache(
    persist_dir=settings.chroma_persist_dir,
    threshold=CACHE_THRESHOLD,
    max_entries=CACHE_MAX_ENTRIES,
)

llm = LLMClient(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    model=settings.llm_model,
)

bm25_retriever = BM25Retriever()

reranker = RerankClient(
    base_url=settings.rerank_llm_base_url,
    api_key=settings.rerank_llm_api_key,
    model=settings.rerank_llm_model,
)

chunker = MarkdownChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)

web_loader = WebLoader(max_depth=20)

from src.tiny_rag.query_log import QueryLog
query_log = QueryLog()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "不支持的文件格式，仅支持 .txt / .md / .pdf"}), 400

    raw: bytes = file.read()
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"

    if ext == ".pdf":
        try:
            content = load_pdf(raw)
        except Exception:
            return jsonify({"error": "无法解析 PDF 文件，请确认文件有效"}), 400
    else:
        content = load_bytes(raw)

    chunks = chunker.chunk_text(content)
    if not chunks:
        return jsonify({"error": "Empty document"}), 400

    embeddings = embedder.embed([c.text for c in chunks])
    vector_store.add_document(doc_id=doc_id, filename=file.filename, chunks=chunks, embeddings=embeddings)
    bm25_retriever.add_document(doc_id=doc_id, filename=file.filename, chunks=[c.text for c in chunks])

    cache.clear()

    return jsonify({"id": doc_id, "filename": file.filename, "chunks": len(chunks)})


@app.route("/upload_web", methods=["POST"])
def upload_web():
    body = request.get_json(silent=True)
    if not body or "url" not in body:
        return jsonify({"error": "Missing 'url' field"}), 400

    url = body["url"]
    max_depth = body.get("max_depth", 20)

    pages = web_loader.load(url, max_depth=max_depth)
    if not pages:
        return jsonify({"error": "No pages could be fetched from the URL"}), 400

    results: list[dict] = []
    for page in pages:
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        chunks = chunker.chunk_text(page.markdown)
        if not chunks:
            continue

        embeddings = embedder.embed([c.text for c in chunks])
        vector_store.add_document(doc_id=doc_id, filename=page.url, chunks=chunks, embeddings=embeddings)
        bm25_retriever.add_document(doc_id=doc_id, filename=page.url, chunks=[c.text for c in chunks])

        results.append({"url": page.url, "chunks": len(chunks)})

    cache.clear()

    return jsonify({"pages": len(results), "results": results})


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True)
    if not body or "question" not in body:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = body["question"]
    force_refresh = body.get("force_refresh", False)
    _t0 = time.time()

    # ── 查询改写 ──
    rewritten = llm.rewrite(question)

    question_vec = embedder.embed([rewritten])[0]

    # ── 语义缓存检查 ──
    if not force_refresh:
        cached = cache.search(query_embedding=question_vec)
        if cached:
            query_log.log_query({
                "original_question": question,
                "rewritten": rewritten,
                "cache_hit": True,
                "latency_ms": round((time.time() - _t0) * 1000),
                "vector_n": VECTOR_N,
                "bm25_n": BM25_N,
                "vector_raw": 0,
                "bm25_raw": 0,
                "final_count": 0,
                "src_vector": 0,
                "src_bm25": 0,
                "src_both": 0,
            })

            def generate_cached():
                yield f"event: context\ndata: {json.dumps(cached['sources'])}\n\n"
                answer = cached["answer"]
                segments = [answer[i:i+3] for i in range(0, len(answer), 3)]
                for seg in segments:
                    yield f"event: token\ndata: {json.dumps(seg)}\n\n"
                yield f"event: done\ndata: {json.dumps({'cached': True, 'original_question': question, 'rewritten': rewritten})}\n\n"

            return Response(
                stream_with_context(generate_cached()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

    # ── 正常检索 + LLM 流程 ──
    # ── 双路检索 + RRF 合并 ──
    # 权重从 data/config.yaml 加载，默认 12/4
    vector_results = vector_store.search(question_vec, n_results=VECTOR_N)
    bm25_results = bm25_retriever.search(question, n_results=BM25_N)

    # 记录两侧的文本集合，用于后续判断最终结果的来源分布
    vector_texts = {r["text"] for r in vector_results}
    bm25_texts = {r["text"] for r in bm25_results}

    results = rrf_merge(vector_results, bm25_results, n_results=10)
    if results and settings.rerank_llm_api_key:
        results = reranker.rerank(question, results, top_n=5)
    elif results:
        results = results[:5]

    # 最终结果的来源分布
    source_dist = {"vector": 0, "bm25": 0, "both": 0}
    for r in results:
        in_v = r["text"] in vector_texts
        in_b = r["text"] in bm25_texts
        if in_v and in_b:
            source_dist["both"] += 1
        elif in_v:
            source_dist["vector"] += 1
        elif in_b:
            source_dist["bm25"] += 1

    if not results:
        return jsonify({"answer": "未找到相关文档，请先上传文档。", "sources": []})

    # 去重后的来源（按 doc_id）
    seen = {}
    for r in results:
        seen.setdefault(r["doc_id"], r.get("filename", r["doc_id"]))
    source_info = [{"id": doc_id, "name": name} for doc_id, name in seen.items()]
    context = "\n\n".join(r["text"] for r in results)

    entry_id = f"cache_{uuid.uuid4().hex[:12]}"

    answer_buffer: list[str] = []

    def generate_and_cache():
        # 1. 推送召回片段
        yield f"event: context\ndata: {json.dumps(results)}\n\n"

        # 2. 逐字推送 LLM token + 收集完整回答
        for token in llm.generate_stream(rewritten, context):
            answer_buffer.append(token)
            yield f"event: token\ndata: {json.dumps(token)}\n\n"

        # 3. 存入缓存
        full_answer = "".join(answer_buffer)
        cache.put(
            question=rewritten,
            answer=full_answer,
            embedding=question_vec,
            sources=results,
            entry_id=entry_id,
        )

        # 4. 记录日志
        query_log.log_query({
            "original_question": question,
            "rewritten": rewritten,
            "cache_hit": False,
            "latency_ms": round((time.time() - _t0) * 1000),
            "vector_n": VECTOR_N,
            "bm25_n": BM25_N,
            "vector_raw": len(vector_results),
            "bm25_raw": len(bm25_results),
            "final_count": len(results),
            "src_vector": source_dist["vector"],
            "src_bm25": source_dist["bm25"],
            "src_both": source_dist["both"],
        })

        # 6. 结束事件
        yield f"event: done\ndata: {json.dumps({'sources': source_info, 'cached': False, 'original_question': question, 'rewritten': rewritten})}\n\n"

    return Response(
        stream_with_context(generate_and_cache()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.route("/documents", methods=["GET"])
def documents():
    return jsonify({"documents": vector_store.list_documents()})


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(cache.get_stats())


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
