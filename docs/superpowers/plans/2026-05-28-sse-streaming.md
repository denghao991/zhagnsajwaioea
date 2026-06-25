# SSE 流式输出 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `/ask` 接口改为 SSE 流式输出，展示召回片段后逐字显示 LLM 回答。

**Architecture:** LLM client 新增 `generate_stream()` 使用 `stream=True`；Flask 路由用 `Response(generator, mimetype="text/event-stream")` 构建 SSE 流；前端用 `fetch` + `ReadableStream` 解析三事件序列（context → token → done）。

**Tech Stack:** 无新增依赖（Flask 原生支持 SSE，openai SDK 已支持 stream）

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/tiny_rag/generation/llm.py` | 修改 | 新增 `generate_stream()` 方法 |
| `src/tiny_rag/app.py` | 修改 | `/ask` 改为 SSE 响应 |
| `src/tiny_rag/templates/index.html` | 修改 | `ask()` 改为流式解析 + 召回片段展示 |
| `tests/test_app.py` | 修改 | 更新 `test_get_documents_empty` 断言（已修复） |

---

### Task 1: LLM 客户端 — 新增流式生成

**Files:**
- Modify: `src/tiny_rag/generation/llm.py`

- [ ] **Step 1: 在 `llm.py` 中新增 `generate_stream()` 方法**

在 `generate()` 方法之后追加：

```python
from collections.abc import Generator


def generate_stream(self, question: str, context: str) -> Generator[str, None, None]:
    """Generate answer tokens via streaming API.

    Yields:
        Each text token as it arrives from the API.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"文档内容：\n{context}\n\n问题：{question}"},
    ]
    response = self._client.chat.completions.create(
        model=self._model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )
    for chunk in response:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token
```

需要将文件顶部的 import 改为：

```python
from collections.abc import Generator

from openai import OpenAI
```

- [ ] **Step 2: 验证导入**

Run: `cd C:\Users\d\PycharmProjects\tiny-rag && python -c "from src.tiny_rag.generation.llm import LLMClient; print('OK')"`

Expected: 打印 "OK"

- [ ] **Step 3: Commit**

```bash
git add src/tiny_rag/generation/llm.py
git commit -m "feat: add stream generate method to LLMClient"
```

---

### Task 2: Flask 路由 — `/ask` 改为 SSE

**Files:**
- Modify: `src/tiny_rag/app.py`

- [ ] **Step 1: 修改 import 和 `/ask` 路由**

文件顶部添加 Flask 的 `Response` 导入（当前只有 `Flask, jsonify, request, render_template`）：

```python
from flask import Flask, Response, jsonify, request, render_template
```

将 `/ask` 路由函数整体替换为：

```python
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

    return Response(generate(), mimetype="text/event-stream")
```

注意：需要添加 `import json` 到文件头部（已有，无需重复添加）。
但 Flask 的 `Response` 需要新增导入。

- [ ] **Step 2: 运行现有测试，验证不破坏已有功能**

Run: `cd C:\Users\d\PycharmProjects\tiny-rag && python -m pytest tests/test_app.py -v --tb=short`

Expected: 8 tests PASS（`test_ask_no_question` 仍返回 400 JSON，不受 SSE 影响）

- [ ] **Step 3: Commit**

```bash
git add src/tiny_rag/app.py
git commit -m "feat: change /ask to SSE streaming response"
```

---

### Task 3: 前端 — 流式解析 SSE + 召回片段展示

**Files:**
- Modify: `src/tiny_rag/templates/index.html`

- [ ] **Step 1: 在 HTML 中添加召回片段展示区域**

在 `.chat-area` 内部，`.messages` 上方添加：

```html
      <div id="context-area" style="display:none; background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:12px; margin-bottom:12px; font-size:13px; color:#666;"></div>
```

- [ ] **Step 2: 重写 `ask()` 函数**

将 `ask()` 函数替换为：

```javascript
async function ask() {
  const input = document.getElementById('question-input');
  const btn = document.getElementById('ask-btn');
  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  btn.disabled = true;
  addMessage(question, 'user');

  const msgDiv = addMessage('', 'assistant');
  const ctxArea = document.getElementById('context-area');
  ctxArea.style.display = 'none';
  ctxArea.innerHTML = '';

  try {
    const resp = await fetch('/ask', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question}) });

    if (!resp.ok) {
      const data = await resp.json();
      msgDiv.textContent = data.error || '请求失败';
      btn.disabled = false;
      return;
    }

    const contentType = resp.headers.get('content-type') || '';
    if (!contentType.includes('event-stream')) {
      const data = await resp.json();
      msgDiv.textContent = data.answer || '未找到相关文档';
      if (data.sources && data.sources.length) {
        msgDiv.innerHTML += '<div class="sources">来源: ' + data.sources.join(', ') + '</div>';
      }
      btn.disabled = false;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentEvent = '';
    let answerBuffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7);
        } else if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (currentEvent === 'context') {
            const chunks = JSON.parse(data);
            ctxArea.innerHTML = '<strong>找到以下相关内容：</strong><br>';
            chunks.forEach(c => {
              ctxArea.innerHTML += '<div style="padding:4px 0;border-bottom:1px solid #f0f0f0;">' + c.text.substring(0, 120) + '...</div>';
            });
            ctxArea.style.display = 'block';
          } else if (currentEvent === 'token') {
            const token = JSON.parse(data);
            answerBuffer += token;
            msgDiv.textContent = answerBuffer;
            msgDiv.scrollIntoView({behavior: 'smooth'});
          } else if (currentEvent === 'done') {
            const meta = JSON.parse(data);
            if (meta.sources && meta.sources.length) {
              msgDiv.innerHTML += '<div class="sources">来源: ' + meta.sources.join(', ') + '</div>';
            }
          }
          currentEvent = '';
        }
      }
    }
  } catch {
    msgDiv.textContent = '网络错误';
  }
  btn.disabled = false;
}
```

- [ ] **Step 3: 验证模板无渲染错误**

Run: `cd C:\Users\d\PycharmProjects\tiny-rag && python -c "from src.tiny_rag.app import app; print('OK')"`

Expected: 打印 "OK"

- [ ] **Step 4: Commit**

```bash
git add src/tiny_rag/templates/index.html
git commit -m "feat: update frontend for SSE streaming and context display"
```

---

### Task 4: 运行完整测试集

**Files:**（无新文件）

- [ ] **Step 1: 运行全部测试**

Run: `cd C:\Users\d\PycharmProjects\tiny-rag && python -m pytest tests/ -v --tb=short`

Expected: 25 tests PASS（`test_ask_no_question` 等 HTTP 逻辑测试不受 SSE 影响）

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "chore: finalize SSE streaming implementation"
```
