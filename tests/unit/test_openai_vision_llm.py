"""Unit tests for OpenAI-compatible Vision LLM initialization."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from src.libs.llm.openai_vision_llm import OpenAIVisionLLM


@dataclass
class MockLLMSettings:
    provider: str = "openai"
    model: str = "qwen3.7-plus"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: str | None = None
    base_url: str | None = None
    azure_endpoint: str | None = None
    api_version: str | None = None


@dataclass
class MockVisionSettings:
    enabled: bool = True
    provider: str = "openai"
    model: str = "qwen-vl-plus"
    max_image_size: int = 2048
    api_key: str | None = None
    base_url: str | None = None
    azure_endpoint: str | None = None
    deployment_name: str | None = None
    api_version: str | None = None


@dataclass
class MockSettings:
    llm: MockLLMSettings
    vision_llm: MockVisionSettings


def test_openai_vision_uses_base_url_from_vision_settings() -> None:
    settings = MockSettings(
        llm=MockLLMSettings(api_version="2024-02-15-preview"),
        vision_llm=MockVisionSettings(
            base_url="https://dashscope.example.com/compatible-mode/v1/",
        ),
    )

    llm = OpenAIVisionLLM(settings, api_key="test-key")

    assert llm.base_url == "https://dashscope.example.com/compatible-mode/v1"
    assert llm._use_azure_auth is False


def test_openai_vision_uses_dashscope_api_key() -> None:
    settings = MockSettings(
        llm=MockLLMSettings(),
        vision_llm=MockVisionSettings(
            base_url="https://dashscope.example.com/compatible-mode/v1",
        ),
    )

    with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "dashscope-key"}, clear=True):
        llm = OpenAIVisionLLM(settings)

    assert llm.api_key == "dashscope-key"
