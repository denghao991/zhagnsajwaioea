"""LLM client — chat via OpenAI-compatible API."""

from collections.abc import Generator

from openai import OpenAI

_SYSTEM_PROMPT = """你是一个基于文档内容回答问题的助手。
请根据以下提供的文档片段来回应用户的问题。
如果你在文档中找不到相关信息，请诚实地说明你不知道。
不要编造信息。"""

_REWRITE_PROMPT = """你是一个RAG系统的问题改写助手。请将用户的口语化问题改写为规范的文档术语表述。

已知术语映射：
{abbreviations}

{pattern}

{few_shot}

要求：
- 保持原意完全不变
- 将缩写替换为全称
- 将检查项名称展开为自然问题
- 仅输出改写后的问题，不要解释，不要加前缀

用户问题：{question}
"""


class LLMClient:
    """Generate answers using OpenAI-compatible chat API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def generate(self, question: str, context: str) -> str:
        """Generate an answer based on question and retrieved context.

        Args:
            question: User's question.
            context: Retrieved document chunks joined as context.

        Returns:
            Generated answer text.
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
        )
        return response.choices[0].message.content or ""

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

    def rewrite(self, question: str) -> str:
        """Normalize user question for better retrieval matching.

        Uses LLM with TERM_MAP abbreviation mapping and few-shot examples
        to expand abbreviations and normalize check-item descriptions.
        """
        from src.tiny_rag.config import TERM_MAP, REWRITE_EXAMPLES, REWRITE_PATTERN

        abbrevs = "\n".join(f"  {k} → {v}" for k, v in TERM_MAP.items())
        few_shot_lines = []
        if REWRITE_EXAMPLES:
            few_shot_lines.append("示例：")
            for ex in REWRITE_EXAMPLES:
                few_shot_lines.append(f"\n问题：{ex['question']}")
                few_shot_lines.append(f"改写：{ex['rewrite']}")
            few_shot_lines.append("")
        few_shot = "\n".join(few_shot_lines)

        prompt = _REWRITE_PROMPT.format(
            abbreviations=abbrevs,
            pattern=REWRITE_PATTERN,
            few_shot=few_shot,
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
