"""No-op and rule-based query rewrite adapters."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol


@dataclass
class QueryRewriteContext:
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    generated_queries: list[str] = field(default_factory=list)
    rewrite_strategy: str = "noop"
    fallback_used: bool = False
    error_message: str = ""
    latency_ms: float = 0.0


class QueryRewriter(Protocol):
    def rewrite(self, query: str, context: QueryRewriteContext | None = None) -> QueryRewriteResult:
        ...


class NoopQueryRewriter:
    def rewrite(self, query: str, context: QueryRewriteContext | None = None) -> QueryRewriteResult:
        return QueryRewriteResult(query, query, [query], "noop")


class RuleBasedQueryRewriter:
    def __init__(self, synonyms: Mapping[str, list[str]] | None = None, abbreviations: Mapping[str, str] | None = None, max_generated_queries: int = 3) -> None:
        self.synonyms = {str(key): [str(item) for item in values] for key, values in (synonyms or {}).items()}
        self.abbreviations = {str(key): str(value) for key, value in (abbreviations or {}).items()}
        self.max_generated_queries = max(1, int(max_generated_queries))

    def rewrite(self, query: str, context: QueryRewriteContext | None = None) -> QueryRewriteResult:
        started = time.monotonic()
        expanded = query
        for key, value in self.abbreviations.items():
            expanded = expanded.replace(key, f"{key} {value}")
        variants = [query]
        for key, values in self.synonyms.items():
            if key in query:
                variants.extend(query.replace(key, value) for value in values)
        if expanded != query:
            variants.append(expanded)
        variants = list(dict.fromkeys(variants))[: self.max_generated_queries]
        return QueryRewriteResult(query, expanded, variants, "rule_based", latency_ms=(time.monotonic() - started) * 1000.0)


class LLMQueryRewriter:
    def __init__(self, rewrite_fn: Callable[[str], str] | None = None, fallback: QueryRewriter | None = None, timeout_seconds: float = 3.0, token_budget: int = 512, cache_enabled: bool = True) -> None:
        self.rewrite_fn = rewrite_fn
        self.fallback = fallback or NoopQueryRewriter()
        self.timeout_seconds = float(timeout_seconds)
        self.token_budget = int(token_budget)
        self.cache_enabled = cache_enabled
        self.cache: dict[str, QueryRewriteResult] = {}

    def rewrite(self, query: str, context: QueryRewriteContext | None = None) -> QueryRewriteResult:
        key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if self.cache_enabled and key in self.cache:
            return self.cache[key]
        if self.rewrite_fn is None:
            result = self.fallback.rewrite(query, context)
            result.fallback_used = True
            result.error_message = "LLM rewrite function is not configured"
            return result
        prompt = query[: max(1, self.token_budget) * 4]
        started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self.rewrite_fn, prompt)
            try:
                rewritten = str(future.result(timeout=self.timeout_seconds) or query)
            except TimeoutError:
                future.cancel()
                raise
            rewritten = rewritten[: max(1, self.token_budget) * 4]
            result = QueryRewriteResult(
                query,
                rewritten,
                list(dict.fromkeys([query, rewritten])),
                "llm",
                latency_ms=(time.monotonic() - started) * 1000.0,
            )
            if self.cache_enabled:
                self.cache[key] = result
            return result
        except TimeoutError:
            result = self.fallback.rewrite(query, context)
            result.fallback_used = True
            result.error_message = "timeout"
            result.latency_ms = (time.monotonic() - started) * 1000.0
            return result
        except Exception as exc:
            result = self.fallback.rewrite(query, context)
            result.fallback_used = True
            result.error_message = type(exc).__name__
            result.latency_ms = (time.monotonic() - started) * 1000.0
            return result
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
