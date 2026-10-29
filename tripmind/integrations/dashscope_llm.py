from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..config import get_config


class DashScopeLLM:
    """Thin OpenAI-compatible client for Alibaba Cloud Model Studio."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install `openai` to use DashScopeLLM.") from exc

        config = dict(config or get_config())
        dashscope = config.get("dashscope", {})

        api_key = api_key or str(dashscope.get("api_key", "")).strip()
        if not api_key:
            raise RuntimeError(
                "No DashScope API key configured. Set dashscope.api_key in "
                "configs/config.local.yaml or DASHSCOPE_API_KEY."
            )

        base_url = (
            base_url
            or dashscope.get("base_url")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model or dashscope.get("model", "qwen-plus")
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
