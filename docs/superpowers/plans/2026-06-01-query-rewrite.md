# LLM 查询改写实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在缓存检查和检索之前，用一次轻量 LLM 调用将用户口语问题改写为规范表述，提高语义缓存命中率和检索精度

**架构：** `LLMClient` 新增 `rewrite()` 方法，`app.py` 的 `/ask` 流中插入改写步骤：原始问题 → LLM 改写 → 改写后问题 → 后续流程全部使用改写后问题

**技术栈：** Python 3.12、OpenAI SDK（已有）

**流程变化：**
```
改写前: 原始问题 → Embedding → 缓存检查 → 检索 → LLM 回答
改写后: 原始问题 → LLM 改写 → 改写后问题 → Embedding → 缓存检查 → 检索 → LLM 回答
                                           ↓
                                     缓存命中直接返回
```

---

### Task 1: 在 LLMClient 中添加 rewrite 方法

**文件:**
- Modify: `src/tiny_rag/generation/llm.py`

- [ ] **Step 1: 编写 rewrite 方法**

```python
_REWRITE_PROMPT = """你是一个问题改写助手。请将用户的口语化问题改写为规范的文档术语表述。
要求：
1. 保持原意完全不变
2. 将口语词替换为业务术语（如"赚了多少"→"营收"、"开头"→"第一季度"）
3. 补全省略成分（如"今年"→"2026年"）
4. 仅输出改写后的问题，不要解释，不要加前缀

用户问题：{question}"""

    def rewrite(self, question: str) -> str:
        """Normalize a user question into standard terminology.

        Uses a cheap LLM call to transform colloquial questions
        into document-friendly terms for better semantic matching.

        Args:
            question: Raw user question.

        Returns:
            Rewritten question in standard terminology.
        """
        messages = [
            {"role": "user", "content": _REWRITE_PROMPT.format(question=question)},
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
            max_tokens=128,
        )
        rewritten = response.choices[0].message.content or question
        return rewritten.strip().strip('"').strip("'")
```

放在 `generate_stream` 方法之后。

- [ ] **Step 2: 运行测试确认通过**

```bash
PYTHONPATH=. python -m pytest tests/test_llm.py -v
```

预期：3 passed

---

### Task 2: 修改 /ask 流程接入改写

**文件:**
- Modify: `src/tiny_rag/app.py`

- [ ] **Step 1: 在 ask() 中 Embedding 之前插入改写**

原始流程：
```python
question = body["question"]
force_refresh = body.get("force_refresh", False)

question_embedding = embedder.embed([question])[0]

# 缓存检查
if not force_refresh:
    cached = cache.search(query_embedding=question_embedding)
    ...
```

改为：
```python
question = body["question"]
force_refresh = body.get("force_refresh", False)

# ── LLM 改写（标准化问题表述）──
rewritten = llm.rewrite(question)

question_embedding = embedder.embed([rewritten])[0]

# ── 语义缓存检查 ──
if not force_refresh:
    cached = cache.search(query_embedding=question_embedding)
    ...
```

- [ ] **Step 2: 将后续所有使用 `question` 的地方改为 `rewritten`（上下文拼接、缓存存储）**

需要改三处：
1. `llm.generate_stream(question, context)` → `llm.generate_stream(rewritten, context)`
2. `cache.put(question=question, ...)` → `cache.put(question=rewritten, ...)`
3. `cache.record_miss(question)` → `cache.record_miss(question)`（保留原始问题，方便了解用户表述）

`cache.record_miss` 和 `missed_questions` 记录原始 `question` 不变，其他全部改为 `rewritten`。

- [ ] **Step 3: 运行测试确认通过**

```bash
PYTHONPATH=. python -m pytest -v
```

预期：34 passed

---

### Task 3: 验证改写效果

**手动物理验证：**

启动应用：
```bash
PYTHONPATH=. python -m src.tiny_rag.app
```

用 curl 模拟口语提问：
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"今年开头赚了多少"}'
```

检查 SSE 返回的 context 是否正确匹配到业务文档。

再次发送相同问题验证缓存命中：
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"今年开头赚了多少"}'
```

检查 `/stats` 确认 `hits` 增加。

用不同措辞验证语义缓存是否命中：
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"一月份营收多少"}'
```

预期：LLM 将两个问题都改写为类似"2026年第一季度营收金额"的规范表述，第二次命中缓存。
