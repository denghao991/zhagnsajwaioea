from pathlib import Path

import yaml
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
    llm_model: str = "deepseek-v4-flash"

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
    "用户问题中的检查项名称格式为'云服务名+检查项描述'，属于风险检查模块。"
    "例如'CSS可用区未多AZ'是一个风险检查项，表示\"云服务CSS的可用区是否未配置多AZ\"。"
    "请改写为'{云服务名}{检查项描述}，这个风险检查项是什么意思？'的格式。"
)

# Few-shot 改写示例（团队持续维护）
# LLM 会参考这些示例来改写用户问题，将常见缩写展开、检查项名称规范化
REWRITE_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "OA有哪些功能",
        "rewrite": "优化顾问(OA)有哪些功能",
    },
    {
        "question": "CSS可用区未多AZ这个是啥意思",
        "rewrite": "云服务CSS可用区未多AZ，这个风险检查项是什么意思？",
    },
]

# 如果外部 YAML 文件存在则覆盖（热加载）
_TERM_MAP_PATH = _PROJECT_ROOT / "data" / "term_map.yaml"
if _TERM_MAP_PATH.exists():
    with open(_TERM_MAP_PATH, encoding="utf-8") as _f:
        _loaded = yaml.safe_load(_f)
        if isinstance(_loaded, dict) and _loaded:
            TERM_MAP = _loaded
