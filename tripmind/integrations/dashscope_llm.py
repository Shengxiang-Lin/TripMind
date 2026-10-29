from __future__ import annotations

import os
from typing import Iterable, Mapping


class DashScopeLLM:
    """Thin OpenAI-compatible client for Alibaba Cloud Model Studio.

    It is intentionally optional: core TripMind runs without this dependency.
    """

    def __init__(self, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install `openai` to use DashScopeLLM.") from exc

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set.")

        base_url = os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model or os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: Iterable[Mapping[str, str]], temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
