"""Alibaba Cloud Bailian / Model Studio LLM implementation.

This provider uses Alibaba Cloud Model Studio's OpenAI-compatible
chat completions endpoint and reuses the existing OpenAI-compatible
request implementation.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError


class BailianLLMError(OpenAILLMError):
    """Raised when Alibaba Cloud Bailian API call fails."""


def _optional_str(value: Any) -> Optional[str]:
    """Return stripped string values, treating blank strings as missing."""
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


class BailianLLM(OpenAILLM):
    """Alibaba Cloud Bailian LLM provider implementation.

    Bailian / Model Studio exposes an OpenAI-compatible chat completions API.
    This provider makes that compatibility explicit in configuration while
    preserving the existing BaseLLM interface.
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Bailian LLM provider.

        Args:
            settings: Application settings containing LLM configuration.
            api_key: Optional API key override. Falls back to settings.llm.api_key
                or the DASHSCOPE_API_KEY environment variable.
            base_url: Optional base URL override. Falls back to settings.llm.base_url
                or the global DashScope compatible-mode endpoint.
            **kwargs: Additional configuration overrides.

        Raises:
            ValueError: If API key is not provided and not found in environment.
        """
        resolved_api_key = (
            _optional_str(api_key)
            or _optional_str(getattr(settings.llm, "api_key", None))
            or _optional_str(os.environ.get("DASHSCOPE_API_KEY"))
        )
        if not resolved_api_key:
            raise ValueError(
                "Alibaba Cloud Bailian API key not provided. Set in settings.yaml "
                "(llm.api_key), DASHSCOPE_API_KEY environment variable, or pass "
                "api_key parameter."
            )

        resolved_base_url = (
            _optional_str(base_url)
            or _optional_str(getattr(settings.llm, "base_url", None))
            or self.DEFAULT_BASE_URL
        )

        super().__init__(
            settings=settings,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            **kwargs,
        )
        self.api_version = None
        self._use_azure_auth = False

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a chat completion using Alibaba Cloud Bailian."""
        try:
            return super().chat(messages=messages, trace=trace, **kwargs)
        except OpenAILLMError as e:
            if isinstance(e, BailianLLMError):
                raise
            message = str(e).replace("[OpenAI]", "[Bailian]")
            raise BailianLLMError(message) from e
