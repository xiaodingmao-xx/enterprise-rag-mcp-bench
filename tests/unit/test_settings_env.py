"""Tests for environment variable expansion in settings YAML."""

from __future__ import annotations

from pathlib import Path
import textwrap

from src.core.settings import load_settings


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _settings_yaml(
    *,
    llm_api_key: str = '"${LLM_API_KEY:-}"',
    embedding_api_key: str = '"${EMBEDDING_API_KEY:-}"',
    vision_api_key: str = '"${VISION_LLM_API_KEY:-}"',
    llm_base_url: str = '"${LLM_BASE_URL:-https://default-llm.example/v1}"',
    max_tokens: str = "1024",
    vision_enabled: str = "true",
) -> str:
    return f"""
    llm:
      provider: openai
      model: gpt-4o-mini
      base_url: {llm_base_url}
      api_key: {llm_api_key}
      temperature: 0.0
      max_tokens: {max_tokens}
      extra_body:
        label: "${{RUNTIME_LABEL:-local}}"
    embedding:
      provider: openai
      model: text-embedding-3-small
      dimensions: 1536
      api_key: {embedding_api_key}
    vision_llm:
      enabled: {vision_enabled}
      provider: openai
      model: gpt-4o-mini
      max_image_size: 2048
      api_key: {vision_api_key}
    vector_store:
      provider: chroma
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: false
      provider: none
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
        - mrr
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    """


def test_load_settings_expands_api_key_environment_variables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "llm-env-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-env-key")
    monkeypatch.setenv("VISION_LLM_API_KEY", "vision-env-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("RUNTIME_LABEL", "ci")

    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, _settings_yaml())

    settings = load_settings(settings_path)

    assert settings.llm.api_key == "llm-env-key"
    assert settings.embedding.api_key == "embedding-env-key"
    assert settings.vision_llm is not None
    assert settings.vision_llm.api_key == "vision-env-key"
    assert settings.llm.base_url == "https://llm.example/v1"
    assert settings.llm.extra_body == {"label": "ci"}


def test_load_settings_uses_default_when_env_is_missing_or_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("EMPTY_LLM_API_KEY", "")

    settings_path = tmp_path / "settings.yaml"
    _write_yaml(
        settings_path,
        _settings_yaml(
            llm_api_key='"${EMPTY_LLM_API_KEY:-fallback-key}"',
            llm_base_url='"${LLM_BASE_URL:-https://fallback.example/v1}"',
        ),
    )

    settings = load_settings(settings_path)

    assert settings.llm.api_key == "fallback-key"
    assert settings.llm.base_url == "https://fallback.example/v1"


def test_missing_env_without_default_expands_to_empty_string(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("MISSING_LLM_API_KEY", raising=False)

    settings_path = tmp_path / "settings.yaml"
    _write_yaml(
        settings_path,
        _settings_yaml(llm_api_key='"${MISSING_LLM_API_KEY}"'),
    )

    settings = load_settings(settings_path)

    assert settings.llm.api_key == ""


def test_unquoted_placeholder_defaults_preserve_yaml_types(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("MAX_TOKENS", raising=False)
    monkeypatch.delenv("VISION_ENABLED", raising=False)

    settings_path = tmp_path / "settings.yaml"
    _write_yaml(
        settings_path,
        _settings_yaml(
            max_tokens="${MAX_TOKENS:-2048}",
            vision_enabled="${VISION_ENABLED:-false}",
        ),
    )

    settings = load_settings(settings_path)

    assert settings.llm.max_tokens == 2048
    assert settings.vision_llm is not None
    assert settings.vision_llm.enabled is False
