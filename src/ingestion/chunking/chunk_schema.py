"""Structured chunk schema helpers.

Concrete chunkers emit :class:`ChunkDraft` objects. ``DocumentChunker`` then
normalises them into the existing ``src.core.types.Chunk`` contract so the rest
of the ingestion pipeline remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re

from src.core.types import Chunk, Document


PageRange = Optional[Tuple[int, int]]


@dataclass
class ChunkDraft:
    """Intermediate structured chunk emitted by chunking strategies."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_index: Optional[int] = None
    source_ref: Optional[str] = None
    page_range: PageRange = None
    section_path: List[str] = field(default_factory=list)
    heading_path: List[str] = field(default_factory=list)
    heading: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    def __post_init__(self) -> None:
        if self.text is None or not str(self.text).strip():
            raise ValueError("Chunk text cannot be empty")
        self.text = str(self.text)


def normalise_string_list(value: Any) -> List[str]:
    """Normalise a metadata value to a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        candidates = re.split(r"[,;\n]+", value)
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = [value]

    result: List[str] = []
    seen = set()
    for item in candidates:
        if isinstance(item, dict):
            raw = item.get("id") or item.get("name") or item.get("text") or ""
        else:
            raw = item
        text = str(raw).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def normalise_page_range(value: Any) -> PageRange:
    """Normalise page range to ``(start, end)`` or ``None``."""
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        start = value.get("start")
        end = value.get("end", start)
    elif isinstance(value, (list, tuple)) and value:
        start = value[0]
        end = value[-1]
    else:
        start = value
        end = value

    try:
        start_int = int(start)
        end_int = int(end)
    except (TypeError, ValueError):
        return None
    if start_int <= 0 or end_int <= 0:
        return None
    if end_int < start_int:
        start_int, end_int = end_int, start_int
    return (start_int, end_int)


def page_range_to_metadata(page_range: PageRange) -> Optional[Dict[str, int]]:
    if page_range is None:
        return None
    return {"start": page_range[0], "end": page_range[1]}


def page_range_to_text(page_range: PageRange) -> str:
    if page_range is None:
        return ""
    return f"{page_range[0]}-{page_range[1]}"


def sync_chunk_metadata(
    *,
    document: Document,
    draft: ChunkDraft,
    chunk_id: str,
    chunk_index: int,
) -> Dict[str, Any]:
    """Merge document/draft metadata and add structured schema fields."""
    metadata: Dict[str, Any] = dict(document.metadata or {})
    doc_images = metadata.pop("images", [])
    metadata.update(draft.metadata or {})

    source_ref = draft.source_ref or document.id
    doc_id = draft.doc_id or document.id
    page_range = draft.page_range or normalise_page_range(metadata.get("page_range"))
    heading_path = draft.heading_path or normalise_string_list(metadata.get("heading_path"))
    section_path = draft.section_path or normalise_string_list(metadata.get("section_path"))
    if not section_path:
        section_path = list(heading_path)

    image_refs = _extract_reference_ids(
        draft.text,
        metadata,
        keys=("image_ids", "image_refs", "images"),
        marker="IMAGE",
    )
    table_ids = _extract_reference_ids(
        draft.text,
        metadata,
        keys=("table_ids", "table_refs", "tables"),
        marker="TABLE",
    )

    chunk_images = _resolve_images(image_refs, doc_images, metadata.get("images"))
    if chunk_images:
        metadata["images"] = chunk_images
    elif "images" in metadata and not metadata["images"]:
        metadata.pop("images", None)

    if chunk_images and metadata.get("page_num") is None:
        metadata["page_num"] = chunk_images[0].get("page")

    metadata.update(
        {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "source_ref": source_ref,
            "char_start": draft.char_start,
            "char_end": draft.char_end,
            "start_offset": draft.char_start,
            "end_offset": draft.char_end,
            "heading_path": heading_path,
            "section_path": section_path,
            "heading": draft.heading or metadata.get("heading"),
            "parent_chunk_id": draft.parent_chunk_id,
            "page_range": page_range_to_metadata(page_range),
            "page_range_text": page_range_to_text(page_range),
            "heading_path_text": " > ".join(heading_path),
            "section_path_text": " > ".join(section_path),
            "image_ids": image_refs,
            "image_refs": image_refs,
            "image_ids_text": ",".join(image_refs),
            "table_ids": table_ids,
            "table_ids_text": ",".join(table_ids),
            "text": draft.text,
        }
    )

    return metadata


def draft_to_chunk(
    *,
    document: Document,
    draft: ChunkDraft,
    chunk_id: str,
    chunk_index: int,
) -> Chunk:
    metadata = sync_chunk_metadata(
        document=document,
        draft=draft,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
    )
    return Chunk(
        id=chunk_id,
        text=draft.text,
        metadata=metadata,
        start_offset=draft.char_start,
        end_offset=draft.char_end,
        source_ref=draft.source_ref or document.id,
        doc_id=draft.doc_id or document.id,
        chunk_index=chunk_index,
        page_range=draft.page_range or normalise_page_range(metadata.get("page_range")),
        section_path=metadata.get("section_path", []),
        heading_path=metadata.get("heading_path", []),
        heading=metadata.get("heading"),
        parent_chunk_id=draft.parent_chunk_id,
        char_start=draft.char_start,
        char_end=draft.char_end,
    )


def _extract_reference_ids(
    text: str,
    metadata: Dict[str, Any],
    *,
    keys: Tuple[str, ...],
    marker: str,
) -> List[str]:
    values: List[Any] = []
    for key in keys:
        item = metadata.get(key)
        if isinstance(item, list):
            values.extend(item)
        elif item:
            values.append(item)
    pattern = rf"\[{marker}:\s*([^\]]+)\]"
    values.extend(re.findall(pattern, text or "", flags=re.IGNORECASE))
    return normalise_string_list(values)


def _resolve_images(
    image_refs: List[str],
    document_images: Any,
    draft_images: Any,
) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    for source in (draft_images, document_images):
        if isinstance(source, list):
            images.extend(item for item in source if isinstance(item, dict))
    if not image_refs:
        return []
    lookup = {str(img.get("id")): img for img in images if img.get("id") is not None}
    return [lookup[img_id] for img_id in image_refs if img_id in lookup]
