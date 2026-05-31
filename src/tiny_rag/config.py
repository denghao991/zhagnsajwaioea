from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
    )

    # GLM (LLM)
    glm_api_key: str = ""
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-4.7"

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
    rerank_api_key: str = ""
    rerank_base_url: str = "https://dashscope.aliyuncs.com"
    rerank_model: str = "gte-rerank"


settings = Settings()
