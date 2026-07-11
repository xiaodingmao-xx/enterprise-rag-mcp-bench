"""Optional contextual retrieval adapters with safe fallbacks."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable, Dict, Protocol


class ChunkContextualizer(Protocol):
    def add_context(self, document_context: str, section_context: str, chunk_text: str) -> str:
        ...


class NoopContextualizer:
    def add_context(self, document_context: str, section_context: str, chunk_text: str) -> str:
        return chunk_text


class RuleBasedContextualizer:
    def __init__(self, max_context_chars: int = 500) -> None:
        self.max_context_chars = max(0, int(max_context_chars))

    def add_context(self, document_context: str, section_context: str, chunk_text: str) -> str:
        parts = [str(item).strip() for item in (document_context, section_context) if str(item).strip()]
        if not parts or self.max_context_chars <= 0:
            return chunk_text
        prefix = "；".join(parts)[: self.max_context_chars]
        return f"{prefix}\n\n{chunk_text}"


class LLMContextualizer:
    def __init__(
        self,
        llm: Callable[[str], str] | None = None,
        fallback: ChunkContextualizer | None = None,
        timeout_seconds: float = 20.0,
        token_budget: int = 512,
        max_estimated_cost_per_document: float = 0.10,
        cache_enabled: bool = True,
    ) -> None:
        self.llm = llm
        self.fallback = fallback or RuleBasedContextualizer()
        self.timeout_seconds = float(timeout_seconds)
        self.token_budget = max(1, int(token_budget))
        self.max_estimated_cost_per_document = float(max_estimated_cost_per_document)
        self.cache_enabled = bool(cache_enabled)
        self.cache: Dict[str, str] = {}
        self.estimated_cost = 0.0
        self.last_fallback = False
        self._last_latency_ms = 0.0

    def add_context(self, document_context: str, section_context: str, chunk_text: str) -> str:
        self.last_fallback = False
        key = hashlib.sha256(f"{document_context}\n{section_context}\n{chunk_text}".encode("utf-8")).hexdigest()
        if self.cache_enabled and key in self.cache:
            return self.cache[key]
        if self.llm is None or self.estimated_cost >= self.max_estimated_cost_per_document:
            self.last_fallback = True
            return self.fallback.add_context(document_context, section_context, chunk_text)
        prompt = f"Document: {document_context}\nSection: {section_context}\nChunk: {chunk_text}"
        prompt = prompt[: self.token_budget * 4]
        started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self.llm, prompt)
            try:
                result = future.result(timeout=self.timeout_seconds)
            except TimeoutError:
                future.cancel()
                raise
            value = str(result or "").strip() or chunk_text
            self.estimated_cost += min(len(prompt) / 4, self.token_budget) / 1000.0
            if self.cache_enabled:
                self.cache[key] = value
            return value
        except TimeoutError:
            self.last_fallback = True
            return self.fallback.add_context(document_context, section_context, chunk_text)
        except Exception:
            self.last_fallback = True
            return self.fallback.add_context(document_context, section_context, chunk_text)
        finally:
            self._last_latency_ms = (time.monotonic() - started) * 1000.0
            executor.shutdown(wait=False, cancel_futures=True)

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms
