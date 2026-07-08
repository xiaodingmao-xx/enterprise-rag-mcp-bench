"""Tests for Bailian/DashScope HTTP reranker."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.libs.reranker.bailian_reranker import (
    BailianRerankError,
    BailianReranker,
)
from src.libs.reranker.reranker_factory import RerankerFactory


class FakeResponse:
    def __init__(self, data, status_code=200, text=""):
        self._data = data
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, endpoint, json, headers, timeout=None):  # noqa: A002, ANN001
        self.calls.append(
            {
                "endpoint": endpoint,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self.response


def _settings(**overrides):
    rerank = {
        "enabled": True,
        "provider": "bailian",
        "model": "qwen3-vl-rerank",
        "api_key": "test-key",
        "endpoint": "https://example.test/api/v1/services/rerank/text-rerank/text-rerank",
        "api_format": "dashscope",
        "timeout_seconds": 12,
        "instruct": None,
        "return_documents": True,
    }
    rerank.update(overrides)
    return SimpleNamespace(
        rerank=SimpleNamespace(**rerank),
        llm=SimpleNamespace(base_url="https://example.test/compatible-mode/v1"),
    )


def test_bailian_reranker_posts_expected_payload_and_sorts_candidates():
    client = FakeClient(
        FakeResponse(
            {
                "output": {
                    "results": [
                        {"index": 2, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.82},
                    ]
                }
            }
        )
    )
    reranker = BailianReranker(settings=_settings(), http_client=client)
    candidates = [
        {"id": "a", "text": "first", "score": 0.1},
        {"id": "b", "text": "second", "score": 0.2},
        {"id": "c", "text": "third", "score": 0.3},
    ]

    result = reranker.rerank("query", candidates)

    assert [item["id"] for item in result] == ["c", "a", "b"]
    assert result[0]["rerank_score"] == pytest.approx(0.91)
    assert result[1]["rerank_score"] == pytest.approx(0.82)
    assert result[2]["rerank_score"] == pytest.approx(0.2)
    assert result[0]["reranker_provider"] == "bailian"

    call = client.calls[0]
    assert call["endpoint"] == _settings().rerank.endpoint
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["json"]["model"] == "qwen3-vl-rerank"
    assert call["json"]["input"]["query"] == "query"
    assert call["json"]["input"]["documents"] == ["first", "second", "third"]
    assert call["json"]["parameters"]["top_n"] == 3
    assert call["json"]["parameters"]["return_documents"] is True


def test_bailian_reranker_raises_on_api_error():
    client = FakeClient(FakeResponse({"message": "bad"}, status_code=500, text="bad"))
    reranker = BailianReranker(settings=_settings(), http_client=client)

    with pytest.raises(BailianRerankError, match="HTTP 500"):
        reranker.rerank("query", [{"id": "a", "text": "first"}])


def test_bailian_reranker_supports_compatible_reranks_payload():
    client = FakeClient(
        FakeResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.97},
                    {"index": 0, "relevance_score": 0.41},
                ]
            }
        )
    )
    settings = _settings(
        model="qwen3-rerank",
        endpoint="https://example.test/compatible-api/v1/reranks",
        api_format="compatible",
        instruct="Given a web search query, retrieve relevant passages.",
    )
    reranker = BailianReranker(settings=settings, http_client=client)
    candidates = [
        {"id": "a", "text": "first", "score": 0.1},
        {"id": "b", "text": "second", "score": 0.2},
    ]

    result = reranker.rerank("query", candidates)

    assert [item["id"] for item in result] == ["b", "a"]
    call = client.calls[0]
    assert call["json"] == {
        "model": "qwen3-rerank",
        "documents": ["first", "second"],
        "query": "query",
        "top_n": 2,
        "instruct": "Given a web search query, retrieve relevant passages.",
    }


def test_bailian_reranker_infers_compatible_format_from_endpoint():
    settings = _settings(
        endpoint="https://example.test/compatible-api/v1/reranks",
        api_format=None,
    )

    reranker = BailianReranker(settings=settings, http_client=FakeClient(FakeResponse({})))

    assert reranker.api_format == "compatible"


def test_bailian_reranker_can_derive_endpoint_from_llm_base_url():
    settings = _settings(endpoint=None)

    reranker = BailianReranker(settings=settings, http_client=FakeClient(FakeResponse({})))

    assert (
        reranker.endpoint
        == "https://example.test/api/v1/services/rerank/text-rerank/text-rerank"
    )


def test_bailian_reranker_can_derive_compatible_endpoint_from_llm_base_url():
    settings = _settings(endpoint=None, api_format="compatible")

    reranker = BailianReranker(settings=settings, http_client=FakeClient(FakeResponse({})))

    assert reranker.endpoint == "https://example.test/compatible-api/v1/reranks"


def test_reranker_factory_registers_bailian_aliases():
    RerankerFactory._PROVIDERS.clear()
    settings = _settings(provider="dashscope")

    reranker = RerankerFactory.create(settings, http_client=FakeClient(FakeResponse({})))

    assert isinstance(reranker, BailianReranker)
    assert "bailian" in RerankerFactory._PROVIDERS
    assert "dashscope" in RerankerFactory._PROVIDERS
