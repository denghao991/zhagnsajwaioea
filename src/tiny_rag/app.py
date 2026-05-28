"""Flask application — RAG system web server."""

import json
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, render_template

from src.tiny_rag.config import settings
from src.tiny_rag.ingestion.loader import load_bytes, load_pdf
from src.tiny_rag.ingestion.chunker import chunk_text
from src.tiny_rag.embedding.client import EmbeddingClient
from src.tiny_rag.storage.vector_store import VectorStore
from src.tiny_rag.generation.llm import LLMClient

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

app = Flask(__name__)

embedder = EmbeddingClient(
    base_url=settings.dashscope_base_url,
    api_key=settings.dashscope_api_key,
    model=settings.embedding_model,
)

vector_store = VectorStore(persist_dir=settings.chroma_persist_dir)

llm = LLMClient(
    base_url=settings.glm_base_url,
    api_key=settings.glm_api_key,
    model=settings.glm_model,
)


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
    vector_store.add_document(doc_id=doc_id, chunks=chunks, embeddings=embeddings)

    return jsonify({"id": doc_id, "filename": file.filename, "chunks": len(chunks)})


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True)
    if not body or "question" not in body:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = body["question"]
    question_embedding = embedder.embed([question])[0]
    results = vector_store.search(question_embedding, n_results=5)

    if not results:
        return jsonify({"answer": "未找到相关文档，请先上传文档。", "sources": []})

    source_ids = list({r["doc_id"] for r in results})
    context = "\n\n".join(r["text"] for r in results)

    def generate():
        # 1. 推送召回片段
        yield f"event: context\ndata: {json.dumps(results)}\n\n"
        # 2. 逐字推送 LLM token
        for token in llm.generate_stream(question, context):
            yield f"event: token\ndata: {json.dumps(token)}\n\n"
        # 3. 结束事件
        yield f"event: done\ndata: {json.dumps({'sources': source_ids})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


@app.route("/documents", methods=["GET"])
def documents():
    return jsonify({"documents": vector_store.list_documents()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
