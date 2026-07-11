"""Optional contextual chunk wrapper."""

from __future__ import annotations

import hashlib
from typing import Any, List

from src.core.types import Document
from src.ingestion.chunking.chunk_schema import ChunkDraft
from src.ingestion.chunking.config import get_chunking_settings, get_section
from src.ingestion.chunking.contextualizer import (
    ChunkContextualizer,
    LLMContextualizer,
    NoopContextualizer,
    RuleBasedContextualizer,
)
from src.ingestion.chunking.recursive_chunker import RecursiveChunker


class ContextualChunker:
    def __init__(self, settings: Any, base: Any = None, contextualizer: ChunkContextualizer | None = None) -> None:
        self.base = base or RecursiveChunker(settings)
        if contextualizer is not None:
            self.contextualizer = contextualizer
            return
        config = get_section(get_chunking_settings(settings), "contextual_retrieval")
        if config.get("enabled", True) is False:
            self.contextualizer = NoopContextualizer()
            return
        provider = str(config.get("provider", "noop")).lower()
        if provider == "rule_based":
            self.contextualizer = RuleBasedContextualizer(config.get("max_context_chars", 500))
        elif provider == "llm":
            llm_config = config.get("llm", {}) or {}
            self.contextualizer = LLMContextualizer(
                llm=None,
                fallback=RuleBasedContextualizer(config.get("max_context_chars", 500)),
                timeout_seconds=llm_config.get("timeout_seconds", 20),
                token_budget=llm_config.get("token_budget", 512),
                max_estimated_cost_per_document=llm_config.get("max_estimated_cost_per_document", 0.10),
                cache_enabled=(config.get("cache", {}) or {}).get("enabled", True),
            )
        else:
            self.contextualizer = NoopContextualizer()

    def split(self, document: Document) -> List[ChunkDraft]:
        drafts = self.base.split(document)
        document_context = str((document.metadata or {}).get("title") or (document.metadata or {}).get("source_type") or "")
        output = []
        for draft in drafts:
            original = draft.text
            contextualized = self.contextualizer.add_context(
                document_context,
                " > ".join(draft.section_path or draft.heading_path),
                original,
            )
            metadata = dict(draft.metadata)
            metadata.update(
                {
                    "original_chunk_text_hash": hashlib.sha256(original.encode("utf-8")).hexdigest(),
                    "contextualized": contextualized != original,
                    "_id_text": original,
                }
            )
            output.append(ChunkDraft(**{**draft.__dict__, "text": contextualized, "metadata": metadata}))
        return output
