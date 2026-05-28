"""LLM client — chat via OpenAI-compatible API."""

from collections.abc import Generator

from openai import OpenAI

_SYSTEM_PROMPT = """你是一个基于文档内容回答问题的助手。
请根据以下提供的文档片段来回应用户的问题。
如果你在文档中找不到相关信息，请诚实地说明你不知道。
不要编造信息。"""


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
