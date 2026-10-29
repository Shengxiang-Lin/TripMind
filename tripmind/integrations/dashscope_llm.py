from __future__ import annotations

import os
from typing import Any, Iterable, Mapping


class DashScopeLLM:
    """Thin OpenAI-compatible client for Alibaba Cloud Model Studio.

    The client supports both ordinary Qwen inference and Alibaba Cloud web
    search. Core TripMind remains runnable without this optional integration.
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

    def chat(
        self,
        messages: Iterable[Mapping[str, str]],
        temperature: float = 0.2,
        *,
        enable_search: bool = False,
        force_search: bool = False,
        search_strategy: str = "turbo",
        extra_body: Mapping[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = dict(extra_body or {})
        if enable_search:
            body["enable_search"] = True
            body["search_options"] = {
                "search_strategy": search_strategy,
                "forced_search": force_search,
            }

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if body:
            kwargs["extra_body"] = body

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
