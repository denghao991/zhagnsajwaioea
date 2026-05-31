"""Flask application — RAG system web server."""

import json
import uuid
from pathlib import Path

from flask import (Flask, Response, jsonify, request, render_template,
                   stream_with_context)

from src.tiny_rag.config import settings
from src.tiny_rag.ingestion.loader import load_bytes, load_pdf
from src.tiny_rag.ingestion.chunker import chunk_text
from src.tiny_rag.embedding.client import EmbeddingClient
from src.tiny_rag.storage.vector_store import VectorStore
from src.tiny_rag.generation.llm import LLMClient
from src.tiny_rag.cache.semantic_cache import SemanticCache
from src.tiny_rag.retrieval.bm25 import BM25Retriever
from src.tiny_rag.retrieval.hybrid import rrf_merge

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

app = Flask(__name__)

embedder = EmbeddingClient(
    base_url=settings.dashscope_base_url,
    api_key=settings.dashscope_api_key,
    model=settings.embedding_model,
)

vector_store = VectorStore(persist_dir=settings.chroma_persist_dir)

cache = SemanticCache(persist_dir=settings.chroma_persist_dir)

llm = LLMClient(
    base_url=settings.glm_base_url,
    api_key=settings.glm_api_key,
    model=settings.glm_model,
)

bm25_retriever = BM25Retriever()


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

    chunks = chunk_text(
        content,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        return jsonify({"error": "Empty document"}), 400

    embeddings = embedder.embed(chunks)
    vector_store.add_document(doc_id=doc_id, filename=file.filename, chunks=chunks, embeddings=embeddings)
    bm25_retriever.add_document(doc_id=doc_id, filename=file.filename, chunks=chunks)

    cache.clear()

    return jsonify({"id": doc_id, "filename": file.filename, "chunks": len(chunks)})


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True)
    if not body or "question" not in body:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = body["question"]
    force_refresh = body.get("force_refresh", False)

    question_embedding = embedder.embed([question])[0]

    # ── 语义缓存检查 ──
    if not force_refresh:
        cached = cache.search(query_embedding=question_embedding)
        if cached and not cached["poisoned"]:
            cache.hits += 1

            def generate_cached():
                yield f"event: context\ndata: {json.dumps(cached['sources'])}\n\n"
                answer = cached["answer"]
                segments = [answer[i:i+3] for i in range(0, len(answer), 3)]
                for seg in segments:
                    yield f"event: token\ndata: {json.dumps(seg)}\n\n"
                yield f"event: done\ndata: {json.dumps({'cached': True})}\n\n"

            return Response(
                stream_with_context(generate_cached()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        elif cached and cached["poisoned"]:
            cache.poisoned_skips += 1
        else:
            cache.misses += 1
            cache.record_miss(question)

    # ── 正常检索 + LLM 流程 ──
    # ── 双路检索 + RRF 合并 ──
    vector_results = vector_store.search(question_embedding, n_results=10)
    bm25_results = bm25_retriever.search(question, n_results=10)
    results = rrf_merge(vector_results, bm25_results, n_results=5)

    if not results:
        return jsonify({"answer": "未找到相关文档，请先上传文档。", "sources": []})

    # 去重后的来源（按 doc_id）
    seen = {}
    for r in results:
        seen.setdefault(r["doc_id"], r.get("filename", r["doc_id"]))
    source_info = [{"id": doc_id, "name": name} for doc_id, name in seen.items()]
    context = "\n\n".join(r["text"] for r in results)

    # 确定 entry_id（force_refresh 时复用已有缓存条目）
    if force_refresh:
        cache.force_refreshes += 1
        found_entry_id = cache.get_entry_id(embedding=question_embedding)
        entry_id = found_entry_id if found_entry_id else f"cache_{uuid.uuid4().hex[:12]}"
    else:
        entry_id = f"cache_{uuid.uuid4().hex[:12]}"

    answer_buffer: list[str] = []

    def generate_and_cache():
        # 1. 推送召回片段
        yield f"event: context\ndata: {json.dumps(results)}\n\n"

        # 2. 逐字推送 LLM token + 收集完整回答
        for token in llm.generate_stream(question, context):
            answer_buffer.append(token)
            yield f"event: token\ndata: {json.dumps(token)}\n\n"

        # 3. 存入缓存
        full_answer = "".join(answer_buffer)
        cache.put(
            question=question,
            answer=full_answer,
            embedding=question_embedding,
            sources=results,
            entry_id=entry_id,
        )

        # 4. force_refresh 时更新 refresh_count
        if force_refresh:
            cache.mark_refreshed(entry_id)

        # 5. 结束事件
        yield f"event: done\ndata: {json.dumps({'sources': source_info})}\n\n"

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
