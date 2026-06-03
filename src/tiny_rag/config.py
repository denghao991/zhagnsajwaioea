from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
    )

    # GLM (LLM)
    llm_api_key: str = ""
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_model: str = "glm-4.7"

    # DashScope / Qwen (Embedding)
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v2"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # DashScope Rerank
    rerank_llm_api_key: str = ""
    rerank_llm_base_url: str = "https://dashscope.aliyuncs.com"
    rerank_llm_model: str = "gte-rerank"


settings = Settings()

# ── 查询改写 ────────────────────────────────────────────
# 缩写 → 全称映射（团队持续维护）
TERM_MAP: dict[str, str] = {
    "OA": "优化顾问(OA)",
    "CSS": "云服务CSS",
    "CCE": "云容器引擎CCE",
}

# 通用检查项推理规则
REWRITE_PATTERN: str = (
    "用户问题可能包含'云服务名+风险描述'格式的检查项名称。"
    "例如'CSS可用区未多AZ'表示云服务CSS在可用区维度上的检查项。"
    "请将其改写为自然的问题描述。"
)
