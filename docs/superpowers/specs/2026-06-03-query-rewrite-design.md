# 查询改写（Query Rewrite）设计

## 目标

在 `app.py` 的 `/ask` 流程中，Embedding 检索之前插入一步 LLM 改写，将用户口语问题改写为文档术语表述，提升语义检索精度。

## 数据流变化

```
改写前: 原始问题 → Embedding → 缓存检查 → 检索 → LLM 回答
改写后: 原始问题 → LLM 改写 → 改写后问题 → Embedding → 缓存检查 → 检索 → LLM 回答
```

仅影响 /ask 流程中 Embedding 前的一步，不改变其他模块。

## 模块设计

### 1. 术语映射 — 写入 `config.py`

不新增文件。在 `src/tiny_rag/config.py` 中增加一个 dict 常量：

```python
# 查询改写用：缩写 → 全称映射
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
```

团队维护时只需编辑这个 dict，新增缩写不需要改代码逻辑。

### 2. `LLMClient.rewrite(question: str) -> str`

新增方法，位于 `llm.py` 中 `generate_stream` 之后。

```python
_REWRITE_PROMPT = """你是一个RAG系统的问题改写助手。请将用户的口语化问题改写为规范的文档术语表述。

已知术语映射：
{abbreviations}

{pattern}

要求：
- 保持原意完全不变
- 将缩写替换为全称
- 将检查项名称展开为自然问题
- 仅输出改写后的问题，不要解释，不要加前缀

用户问题：{question}
"""

def rewrite(self, question: str) -> str:
    """Normalize user question for better retrieval matching.

    Expands abbreviations and check-item names into document-friendly terms.
    Falls back to original question on any failure.
    """
    from src.tiny_rag.config import TERM_MAP, REWRITE_PATTERN

    abbrevs = "\n".join(f"  {k} → {v}" for k, v in TERM_MAP.items())
    prompt = _REWRITE_PROMPT.format(
        abbreviations=abbrevs,
        pattern=REWRITE_PATTERN,
        question=question,
    )
    try:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=128,
        )
        rewritten = resp.choices[0].message.content or question
        return rewritten.strip().strip('"').strip("'")
    except Exception:
        return question
```

### 3. `/ask` 流程改动

文件：`src/tiny_rag/app.py`

改动三处：

| 位置 | 原代码 | 改为 |
|---|---|---|
| Embedding 前 (原第 140 行) | `question_embedding = embedder.embed([question])[0]` | `question_embedding = embedder.embed([rewritten])[0]` |
| LLM 生成 (原第 203 行) | `llm.generate_stream(question, context)` | `llm.generate_stream(rewritten, context)` |
| 缓存存储 (原第 210 行) | `cache.put(question=question, ...)` | `cache.put(question=rewritten, ...)` |

同时新增改写步骤：

```python
question = body["question"]
force_refresh = body.get("force_refresh", False)

# ── 查询改写 ──
rewritten = llm.rewrite(question)

question_embedding = embedder.embed([rewritten])[0]
```

`cache.record_miss(question)` 保留原始问题不变。

### 4. 错误处理

| 场景 | 行为 |
|---|---|
| LLM API 调用失败 | 返回原始 `question`，继续后续流程 |
| LLM 返回空字符串 | 返回原始 `question` |
| `TERM_MAP` 为空 | 仍然正常改写，只走通用规则 |

## 文件清单

| 操作 | 文件 | 说明 |
|---|---|---|
| 修改 | `src/tiny_rag/config.py` | 新增 `TERM_MAP` 和 `REWRITE_PATTERN` |
| 修改 | `src/tiny_rag/generation/llm.py` | 新增 `rewrite()` 方法 |
| 修改 | `src/tiny_rag/app.py` | /ask 流程接入改写 |
| 修改 | `tests/test_llm.py` | 新增 rewrite 测试 |

## 测试策略

- `test_rewrite_expands_abbreviation`：mock LLM 返回"优化顾问(OA)有哪些功能"，验证返回正确
- `test_rewrite_fallback`：mock LLM 抛异常，验证返回原始 question
- `test_term_map_exists`：验证 `TERM_MAP` 包含预期键值
