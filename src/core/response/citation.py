"""Structured citation records shared by answer and verification stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class CitationRecord:
    """A citation tied to one chunk in the current retrieval context.

    The legacy ``source``/``page``/``snippet`` aliases are intentionally kept
    in :meth:`to_dict` so existing MCP consumers remain compatible.
    """

    citation_id: str
    document_id: Optional[str]
    version_id: Optional[str]
    chunk_id: str
    source_uri: Optional[str]
    source_title: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    quoted_span: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context(cls, context: Any, *, max_chars: int = 240) -> "CitationRecord":
        metadata = dict(getattr(context, "metadata", {}) or {})
        return cls._from_values(
            citation_id=getattr(context, "citation_id", None),
            chunk_id=getattr(context, "chunk_id", ""),
            text=getattr(context, "text", ""),
            score=getattr(context, "score", 0.0),
            source=getattr(context, "source", None),
            page=getattr(context, "page", None),
            metadata=metadata,
            max_chars=max_chars,
        )

    @classmethod
    def from_retrieval_result(
        cls,
        result: Any,
        *,
        citation_id: Optional[str] = None,
        max_chars: int = 240,
    ) -> "CitationRecord":
        metadata = dict(getattr(result, "metadata", {}) or {})
        if getattr(result, "document_id", None):
            metadata.setdefault("document_id", getattr(result, "document_id"))
        if getattr(result, "version_id", None):
            metadata.setdefault("version_id", getattr(result, "version_id"))
        if getattr(result, "page_number", None) is not None:
            metadata.setdefault("page_number", getattr(result, "page_number"))
        if getattr(result, "source", None):
            metadata.setdefault("source", getattr(result, "source"))
        return cls._from_values(
            citation_id=citation_id,
            chunk_id=getattr(result, "chunk_id", ""),
            text=getattr(result, "text", ""),
            score=getattr(result, "score", 0.0),
            source=(
                getattr(result, "source", None)
                or metadata.get("source_uri")
                or metadata.get("source_path")
                or metadata.get("source")
            ),
            page=getattr(result, "page_number", None),
            metadata=metadata,
            max_chars=max_chars,
        )

    @classmethod
    def _from_values(
        cls,
        *,
        citation_id: Optional[str],
        chunk_id: Any,
        text: Any,
        score: Any,
        source: Any,
        page: Any,
        metadata: dict[str, Any],
        max_chars: int,
    ) -> "CitationRecord":
        document_id = _first_text(
            metadata.get("document_id"),
            metadata.get("doc_id"),
            metadata.get("source_doc_id"),
        )
        version_id = _first_text(metadata.get("version_id"))
        source_uri = _first_text(
            metadata.get("source_uri"),
            metadata.get("source_path"),
            metadata.get("source"),
            source,
        )
        source_title = _first_text(metadata.get("source_title"), metadata.get("title"))
        page_start = _coerce_int(
            metadata.get("page_start"),
            metadata.get("page_num"),
            metadata.get("page_number"),
            metadata.get("page"),
            page,
        )
        page_end = _coerce_int(metadata.get("page_end"), page_start)
        try:
            confidence = max(0.0, min(1.0, float(score or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            citation_id=str(citation_id or "C1"),
            document_id=document_id,
            version_id=version_id,
            chunk_id=str(chunk_id or ""),
            source_uri=source_uri,
            source_title=source_title,
            page_start=page_start,
            page_end=page_end,
            quoted_span=_snippet(str(text or ""), max_chars),
            confidence=confidence,
            metadata=metadata,
        )

    @property
    def source(self) -> Optional[str]:
        return self.source_uri

    @property
    def page(self) -> Optional[int]:
        return self.page_start

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Backward-compatible aliases used by the existing MCP response.
        payload["source"] = self.source_uri
        payload["page"] = self.page_start
        payload["snippet"] = self.quoted_span
        payload["source_path"] = self.source_uri
        return payload


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _coerce_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None or str(value).strip() == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _snippet(text: str, max_chars: int) -> str:
    limit = max(1, int(max_chars or 240))
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "..."
