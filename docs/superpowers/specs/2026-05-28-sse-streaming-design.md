# SSE 流式输出 — 设计文档

**日期：** 2026-05-28
**项目：** Tiny RAG

## 概述

将 `/ask` 接口从同步 JSON 响应改为 SSE（Server-Sent Events）流式输出。用户提问后，页面先展示检索到的相关片段，然后 LLM 答案逐字输出。

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| SSE 事件结构 | `context` → `token` → `done` 三事件序列 | 清晰、职责单一、前端易解析 |
| 依赖 | 无新增 | `openai` SDK 已支持 `stream=True`，Flask 原生支持 `Response(generator, mimetype="text/event-stream")` |

## SSE 事件流

```
event: context
data: [{"doc_id": "doc_001", "text": "..."}, ...]

event: token
data: "根"

event: token
data: "据"

...

event: done
data: {"sources": ["doc_001", "doc_002"]}
```

## 变更清单

### 1. `generation/llm.py`

新增流式方法：

```python
from collections.abc import Generator

def generate_stream(self, question: str, context: str) -> Generator[str, None, None]:
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

保留原有的 `generate()` 方法不变（后续可清理，本次不做）。

### 2. `app.py`

`/ask` 路由改为 SSE 端点：

- 检索逻辑与现有相同
- 嵌入 + 检索 → 返回 200 SSE 流
- 无检索结果 → 直接返回 `{"answer": "未找到...", "sources": []}`（同现有）
- 检索成功 → 返回 SSE 流，顺序：
  1. `context` 事件：召回片段列表（含 doc_id + text）
  2. `token` 事件：逐字推送 LLM token
  3. `done` 事件：sources（文档 ID 列表）

### 3. `templates/index.html`

`ask()` 函数改用 `fetch` + `ReadableStream` 解析 SSE：

- 发送 POST 请求到 `/ask`
- 解析 `event:` / `data:` 行
- `context` 事件：在消息框上方展示召回片段
- `token` 事件：追加到占位消息元素
- `done` 事件：追加 sources 信息

### 4. 测试

- `tests/test_llm.py`：增加 `test_generate_stream_returns_tokens()`（需要 API key，可跳过）
- `tests/test_app.py`：修改现有 `/ask` 测试适配 SSE 响应，或增加 SSE 流式测试

## 不涉及

- 上传、文档列表接口保持不变（同步 JSON）
- 不修改向量存储、分块、嵌入逻辑
- 不引入额外 SSE 库

## 边界情况

| 场景 | 行为 |
|------|------|
| 无检索结果 | 直接返回 JSON（非 SSE），保持现有行为 |
| LLM 流式中断 | SSE 连接断开，前端展示已收到的内容 |
| LLM 返回空 token | `stream=True` 下 chunk 可能空内容，只 yield 非空 token |
| 并发请求 | Flask 开发服务器为同步模型，每个 SSE 请求占用一个 worker；生产环境需用 WSGI 异步服务器（后续优化） |
