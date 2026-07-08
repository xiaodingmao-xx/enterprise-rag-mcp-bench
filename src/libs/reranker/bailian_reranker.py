"""Alibaba Cloud Bailian/DashScope reranker implementation."""

from __future__ import annotations

import os
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker


class BailianRerankError(RuntimeError):
    """Raised when Bailian reranking fails."""


class BailianReranker(BaseReranker):
    """Rerank candidates through Bailian's text-rerank HTTP API."""

    DEFAULT_RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"
    DEFAULT_COMPATIBLE_RERANK_PATH = "/compatible-api/v1/reranks"
    API_FORMAT_DASHSCOPE = "dashscope"
    API_FORMAT_COMPATIBLE = "compatible"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: Optional[float] = None,
        http_client: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.model = self._get_model_name(settings)
        self.api_key = self._resolve_api_key(settings, api_key)
        configured_endpoint = self._resolve_configured_endpoint(settings, endpoint)
        self.api_format = self._resolve_api_format(settings, configured_endpoint)
        self.endpoint = self._resolve_endpoint(
            settings,
            configured_endpoint,
            self.api_format,
        )
        self.timeout = self._resolve_timeout(settings, timeout)
        self.instruct = self._optional_str(
            getattr(getattr(settings, "rerank", None), "instruct", None)
        )
        self.return_documents = bool(
            getattr(getattr(settings, "rerank", None), "return_documents", True)
        )
        self.http_client = http_client
        self.kwargs = kwargs

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)

        top_n = self._resolve_top_n(candidates, kwargs)
        documents = [self._candidate_text(candidate) for candidate in candidates]
        payload = self._build_payload(
            query=query,
            documents=documents,
            top_n=top_n,
        )

        data = self._post_payload(payload)
        results = self._extract_results(data)
        return self._map_results_to_candidates(results, candidates)

    def _build_payload(
        self,
        *,
        query: str,
        documents: List[str],
        top_n: int,
    ) -> Dict[str, Any]:
        if self.api_format == self.API_FORMAT_COMPATIBLE:
            payload: Dict[str, Any] = {
                "model": self.model,
                "documents": documents,
                "query": query,
                "top_n": top_n,
            }
            if self.instruct:
                payload["instruct"] = self.instruct
            return payload

        if self.api_format == self.API_FORMAT_DASHSCOPE:
            return {
                "model": self.model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "return_documents": self.return_documents,
                    "top_n": top_n,
                },
            }

        raise BailianRerankError(
            f"Unsupported Bailian rerank api_format: {self.api_format!r}"
        )

    def _post_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.http_client is not None:
            response = self._client_post(
                self.http_client,
                self.endpoint,
                payload,
                headers,
            )
        else:
            try:
                import httpx
            except ImportError as exc:
                raise BailianRerankError(
                    "httpx is required for Bailian reranking"
                ) from exc

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.endpoint,
                        json=payload,
                        headers=headers,
                    )
            except Exception as exc:
                raise BailianRerankError(f"Bailian rerank request failed: {exc}") from exc

        status_code = getattr(response, "status_code", 200)
        if int(status_code) >= 400:
            detail = getattr(response, "text", "")
            raise BailianRerankError(
                f"Bailian rerank API error HTTP {status_code}: {detail}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise BailianRerankError(
                f"Bailian rerank response is not valid JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise BailianRerankError(
                f"Bailian rerank response must be an object, got {type(data).__name__}"
            )
        return data

    def _client_post(
        self,
        client: Any,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Any:
        try:
            return client.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except TypeError:
            return client.post(endpoint, json=payload, headers=headers)
        except Exception as exc:
            raise BailianRerankError(f"Bailian rerank request failed: {exc}") from exc

    def _extract_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        output = data.get("output", data)
        if not isinstance(output, dict):
            raise BailianRerankError("Bailian rerank response missing object output")

        results = output.get("results", data.get("results"))
        if not isinstance(results, list):
            raise BailianRerankError("Bailian rerank response missing results list")

        return [item for item in results if isinstance(item, dict)]

    def _map_results_to_candidates(
        self,
        results: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        reranked: List[Dict[str, Any]] = []
        seen_indices: set[int] = set()

        for fallback_index, item in enumerate(results):
            index = self._parse_result_index(item, fallback_index)
            if index < 0 or index >= len(candidates) or index in seen_indices:
                continue

            candidate = candidates[index].copy()
            candidate["rerank_score"] = self._parse_score(item, candidate)
            candidate["reranker_provider"] = "bailian"
            reranked.append(candidate)
            seen_indices.add(index)

        for index, candidate in enumerate(candidates):
            if index in seen_indices:
                continue
            fallback = candidate.copy()
            fallback["rerank_score"] = float(candidate.get("score", 0.0) or 0.0)
            fallback["reranker_provider"] = "bailian"
            reranked.append(fallback)

        return reranked

    @staticmethod
    def _parse_result_index(item: Dict[str, Any], fallback_index: int) -> int:
        raw_index = item.get("index", item.get("document_index", fallback_index))
        try:
            return int(raw_index)
        except (TypeError, ValueError):
            return fallback_index

    @staticmethod
    def _parse_score(item: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        for key in ("relevance_score", "score", "rerank_score"):
            if key in item:
                try:
                    return float(item[key])
                except (TypeError, ValueError):
                    break
        return float(candidate.get("score", 0.0) or 0.0)

    @staticmethod
    def _candidate_text(candidate: Dict[str, Any]) -> str:
        text = candidate.get("text") or candidate.get("content") or ""
        return str(text)

    @staticmethod
    def _get_model_name(settings: Any) -> str:
        model = getattr(getattr(settings, "rerank", None), "model", None)
        if not isinstance(model, str) or not model.strip():
            raise BailianRerankError("Missing rerank.model for Bailian reranker")
        return model.strip()

    @classmethod
    def _resolve_api_key(cls, settings: Any, api_key: Optional[str]) -> str:
        resolved = (
            cls._optional_str(api_key)
            or cls._optional_str(getattr(getattr(settings, "rerank", None), "api_key", None))
            or cls._optional_str(os.environ.get("DASHSCOPE_API_KEY"))
            or cls._optional_str(os.environ.get("BAILIAN_API_KEY"))
        )
        if not resolved:
            raise BailianRerankError(
                "Bailian rerank API key not provided. Set rerank.api_key or "
                "DASHSCOPE_API_KEY."
            )
        return resolved

    @classmethod
    def _resolve_configured_endpoint(
        cls,
        settings: Any,
        endpoint: Optional[str],
    ) -> Optional[str]:
        return (
            cls._optional_str(endpoint)
            or cls._optional_str(getattr(getattr(settings, "rerank", None), "endpoint", None))
        )

    @classmethod
    def _resolve_endpoint(
        cls,
        settings: Any,
        configured_endpoint: Optional[str],
        api_format: str,
    ) -> str:
        if configured_endpoint:
            return configured_endpoint

        llm_base_url = cls._optional_str(getattr(getattr(settings, "llm", None), "base_url", None))
        if llm_base_url:
            parsed = urlparse(llm_base_url)
            if parsed.scheme and parsed.netloc:
                path = (
                    cls.DEFAULT_COMPATIBLE_RERANK_PATH
                    if api_format == cls.API_FORMAT_COMPATIBLE
                    else cls.DEFAULT_RERANK_PATH
                )
                return f"{parsed.scheme}://{parsed.netloc}{path}"

        raise BailianRerankError(
            "Bailian rerank endpoint not provided. Set rerank.endpoint."
        )

    @classmethod
    def _resolve_api_format(
        cls,
        settings: Any,
        configured_endpoint: Optional[str],
    ) -> str:
        configured = cls._optional_str(
            getattr(getattr(settings, "rerank", None), "api_format", None)
        )
        if configured:
            normalised = configured.lower()
        elif configured_endpoint and (
            "/compatible-api/" in configured_endpoint
            or configured_endpoint.rstrip("/").endswith("/reranks")
        ):
            normalised = cls.API_FORMAT_COMPATIBLE
        else:
            normalised = cls.API_FORMAT_DASHSCOPE

        aliases = {
            "dashscope": cls.API_FORMAT_DASHSCOPE,
            "text-rerank": cls.API_FORMAT_DASHSCOPE,
            "service": cls.API_FORMAT_DASHSCOPE,
            "compatible": cls.API_FORMAT_COMPATIBLE,
            "openai": cls.API_FORMAT_COMPATIBLE,
            "reranks": cls.API_FORMAT_COMPATIBLE,
        }
        resolved = aliases.get(normalised, normalised)
        if resolved not in {cls.API_FORMAT_DASHSCOPE, cls.API_FORMAT_COMPATIBLE}:
            raise BailianRerankError(
                "Unsupported rerank.api_format. Expected 'dashscope' or "
                f"'compatible', got {configured!r}."
            )
        return resolved

    @staticmethod
    def _resolve_timeout(settings: Any, timeout: Optional[float]) -> float:
        if isinstance(timeout, (int, float)) and timeout > 0:
            return float(timeout)
        configured = getattr(
            getattr(settings, "rerank", None),
            "timeout_seconds",
            None,
        )
        if isinstance(configured, (int, float)) and configured > 0:
            return float(configured)
        return 30.0

    @staticmethod
    def _resolve_top_n(candidates: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> int:
        raw_top_n = kwargs.get("top_n", kwargs.get("top_k", len(candidates)))
        try:
            top_n = int(raw_top_n)
        except (TypeError, ValueError):
            top_n = len(candidates)
        return max(1, min(top_n, len(candidates)))

    @staticmethod
    def _optional_str(value: Any) -> Optional[str]:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return None
