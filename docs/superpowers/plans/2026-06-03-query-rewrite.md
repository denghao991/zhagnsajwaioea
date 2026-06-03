# 查询改写（Query Rewrite）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 /ask 流程中 Embedding 检索前插入 LLM 改写，将口语问题（含缩写/检查项名）转为文档术语，提升检索精度

**Architecture:** `config.py` 新增 `TERM_MAP` dict + `REWRITE_PATTERN` 字符串；`LLMClient.rewrite()` 用它们构建 prompt 调用 LLM；`app.py` 的 `/ask` 流程在 Embedding 前插入 `rewritten = llm.rewrite(question)`，后续全部使用 `rewritten`

**Tech Stack:** Python 3.12, OpenAI SDK (已有), pytest + mock

---

### Task 1: config.py 新增术语映射

**Files:**
- Modify: `src/tiny_rag/config.py`

- [ ] **Step 1: 在 config.py 末尾添加 TERM_MAP 和 REWRITE_PATTERN**

```python
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
```

- [ ] **Step 2: 验证 import 可用**

Run: `python -c "from src.tiny_rag.config import TERM_MAP, REWRITE_PATTERN; print(TERM_MAP)"`
Expected: `{'OA': '优化顾问(OA)', 'CSS': '云服务CSS', 'CCE': '云容器引擎CCE'}`

- [ ] **Step 3: Commit**

```bash
git add src/tiny_rag/config.py
git commit -m "feat: add term map and rewrite pattern for query rewriting"
```

---

### Task 2: LLMClient 新增 rewrite() 方法

**Files:**
- Modify: `src/tiny_rag/generation/llm.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: 编写 rewrite 方法的测试**

在 `tests/test_llm.py` 末尾追加：

```python
def test_rewrite_expands_abbreviation():
    client = LLMClient(base_url="http://fake", api_key="test", model="test-model")

    mock_message = MagicMock()
    mock_message.content = "优化顾问(OA)有哪些功能"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(client._client.chat.completions, "create", return_value=mock_response):
        result = client.rewrite("OA有哪些功能")

    assert result == "优化顾问(OA)有哪些功能"


def test_rewrite_fallback_on_error():
    client = LLMClient(base_url="http://fake", api_key="test", model="test-model")

    with patch.object(client._client.chat.completions, "create", side_effect=Exception("API error")):
        result = client.rewrite("测试问题")

    assert result == "测试问题"


def test_rewrite_fallback_on_empty():
    client = LLMClient(base_url="http://fake", api_key="test", model="test-model")

    mock_message = MagicMock()
    mock_message.content = None
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(client._client.chat.completions, "create", return_value=mock_response):
        result = client.rewrite("测试问题")

    assert result == "测试问题"
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `pytest tests/test_llm.py::test_rewrite_expands_abbreviation tests/test_llm.py::test_rewrite_fallback_on_error tests/test_llm.py::test_rewrite_fallback_on_empty -v`
Expected: 3 FAIL (method not defined)

- [ ] **Step 3: 实现 rewrite() 方法**

在 `src/tiny_rag/generation/llm.py` 的 `generate_stream` 方法之后添加：

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

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/test_llm.py::test_rewrite_expands_abbreviation tests/test_llm.py::test_rewrite_fallback_on_error tests/test_llm.py::test_rewrite_fallback_on_empty -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/tiny_rag/generation/llm.py tests/test_llm.py
git commit -m "feat: add LLMClient.rewrite() for query normalization"
```

---

### Task 3: /ask 流程接入查询改写

**Files:**
- Modify: `src/tiny_rag/app.py`

- [ ] **Step 1: 在 ask() 中 Embedding 前插入 rewrite 步骤**

找到 `src/tiny_rag/app.py:140` 附近：

```python
    question = body["question"]
    force_refresh = body.get("force_refresh", False)

    question_embedding = embedder.embed([question])[0]
```

改为：

```python
    question = body["question"]
    force_refresh = body.get("force_refresh", False)

    # ── 查询改写 ──
    rewritten = llm.rewrite(question)

    question_embedding = embedder.embed([rewritten])[0]
```

- [ ] **Step 2: 后续流程将 question 改为 rewritten**

第 203 行附近 `llm.generate_stream(question, context)` → `llm.generate_stream(rewritten, context)`

第 209 行附近 `cache.put(question=question, ...)` → `cache.put(question=rewritten, ...)`

- [ ] **Step 3: 确认 record_miss 保留原始问题**

找到 `cache.record_miss(question)` — 确认它保持不变（记录用户原始问法）

- [ ] **Step 4: 运行现有测试，确认未引入回归**

Run: `pytest tests/test_app.py -v`
Expected: 所有 test_ask_* 测试都通过

- [ ] **Step 5: Commit**

```bash
git add src/tiny_rag/app.py
git commit -m "feat: integrate query rewrite into /ask flow"
```

---

### Task 4: 验证改写效果（人工验证）

- [ ] **Step 1: 启动应用** — `python -m src.tiny_rag.app`

- [ ] **Step 2: 上传测试文档并测试改写**

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"OA有哪些功能"}'
```

验证 SSE 返回的 context 包含产品介绍#0 的 chunk

- [ ] **Step 3: 测试检查项问法**

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"CSS可用区未多AZ这个是啥意思"}'
```

验证能正常检索到相关文档

- [ ] **Step 4: 测试错误降级**

杀掉网络或改错 API key，确认 rewrite 降级返回原始问题，不影响正常检索
