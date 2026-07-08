"""
文档处理 —— MarkdownHeaderTextSplitter + 离线索引构建

核心认知：
- 用 MarkdownHeaderTextSplitter 按标题层级切分，保留标题路径
- Chunk size 选型：256 截断率高，384/512 对比消融，384 最终胜出
- 标题路径（如"OA > 监控告警 > 配置步骤"）拼入 chunk content 头部
- 特殊文本（表格/代码块/JSON）用 LLM 生成自然语言摘要

API 要点（面试用）：
- LangChain MarkdownHeaderTextSplitter
- headers_to_split_on 定义切分标题层级
- chunk overlaps 0（语义完整切分，不需要 overlap）
"""

from langchain.text_splitter import MarkdownHeaderTextSplitter


# --- 标题层级切分配置 ---
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Chunk size 选型数据（Q3）
CHUNK_SIZE_EXPERIMENT = {
    "256": {"截断率": "偏高，频繁腰斩语义", "MRR": 0.794, "Recall@5": 0.861},
    "384": {"截断率": "低，~1%，语义完整",     "MRR": 0.831, "Recall@5": 0.892, "结论": "✅ 最终选型"},
    "512": {"截断率": "低，~1%，但token浪费",  "MRR": 0.833, "Recall@5": 0.894, "结论": "指标略高但token多33%"},
}

CHUNK_OVERLAP = 0  # 标题自然边界切分，不需要重叠


# --- 文档切分流程 ---
def process_documents(markdown_files: list[str]) -> list[dict]:
    """
    离线文档处理入口

    流程：读取 Markdown → MarkdownHeaderTextSplitter 切分
         → 标题路径拼入 chunk content → LLM 特殊文本摘要（可选）
         → text-embedding-v3 向量化 → ChromaDB 写入

    Args:
        markdown_files: Markdown 文件路径列表

    Returns:
        chunks: [{"id": ..., "content": ..., "metadata": {...}}, ...]
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # 保留标题行本身，不删掉
    )

    all_chunks = []
    for file_path in markdown_files:
        with open(file_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        # 按标题层级切分
        docs = splitter.split_text(markdown_text)

        for i, doc in enumerate(docs):
            # 构建标题路径（从 metadata 中提取）
            headers_path = []
            for level in ["h1", "h2", "h3"]:
                if doc.metadata.get(level):
                    headers_path.append(doc.metadata[level])
            header_prefix = " > ".join(headers_path)

            # 标题路径拼入 chunk content 头部（提升检索效果）
            full_content = f"{header_prefix}\n{doc.page_content}"

            # 特殊文本摘要：表格/代码块/JSON 用 LLM 生成自然语言描述
            if _is_special_text(doc.page_content):
                summary = _generate_llm_summary(doc.page_content)
                full_content = f"{full_content}\n（摘要：{summary}）"

            chunk_id = f"{file_path}_chunk_{i}"

            all_chunks.append({
                "id": chunk_id,
                "content": full_content,
                "metadata": {
                    "file_name": file_path,
                    "headers": header_prefix,
                    "chunk_index": i,
                    # ChromaDB 只支持 str/int/float/bool 元数据
                }
            })

    return all_chunks


def _is_special_text(text: str) -> bool:
    """判断是否为需要 LLM 摘要的特殊文本"""
    # 表格、代码块、JSON 配置 —— 纯原始数据，自然语言少，摘要帮助检索
    text_stripped = text.strip()
    return (
        text_stripped.startswith("|") or      # Markdown 表格
        text_stripped.startswith("```") or    # 代码块
        text_stripped.startswith("{") or      # JSON
        text_stripped.startswith("<!--")       # HTML 注释
    )


def _generate_llm_summary(special_text: str) -> str:
    """用 LLM 为特殊文本生成自然语言摘要"""
    # prompt = f"用一句话描述以下技术内容：\n{special_text}"
    # return llm_client.generate(prompt, max_tokens=50)
    pass


# --- 文档更新流程（Q4）---
def update_document(file_path: str):
    """
    文档更新（全量删除 + 重新插入）

    为什么不是逐 chunk 对齐更新？
    - 文档内容修改后（增/删章节），chunk 边界整体漂移
    - Old[3] 和 New[3] 对应的是完全不同的内容 → chunk_index 对齐没有意义
    - 正确的做法：删掉旧文档的所有 chunk → 重新切分 → 重新 embedding → 重新写入
    """
    # 1. 删除旧 chunk（按 file_name 过滤）
    collection.delete(where={"file_name": file_path})
    # 2. 重新切分
    new_chunks = process_documents([file_path])
    # 3. 重新 embedding + 写入
    for chunk in new_chunks:
        collection.add(
            ids=[chunk["id"]],
            documents=[chunk["content"]],
            metadatas=[chunk["metadata"]],
        )
    # 4. 重建 BM25 倒排索引（从 ChromaDB 读全量 chunk 重建，< 1s）
    rebuild_bm25_index()
