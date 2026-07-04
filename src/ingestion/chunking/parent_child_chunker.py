"""Parent-child hierarchical chunker."""

from __future__ import annotations

from typing import Any, List

from src.core.types import Document
from src.ingestion.chunking.chunk_id import generate_chunk_id, resolve_doc_hash
from src.ingestion.chunking.chunk_schema import ChunkDraft
from src.ingestion.chunking.config import get_chunking_settings, get_section
from src.ingestion.chunking.recursive_chunker import RecursiveChunker


class ParentChildChunker:
    """Generate parent context chunks and child retrieval chunks."""

    def __init__(self, settings: Any) -> None:
        chunking = get_chunking_settings(settings)
        config = get_section(chunking, "parent_child")
        self.parent_chunk_size = int(config.get("parent_chunk_size") or 1600)
        self.parent_chunk_overlap = int(config.get("parent_chunk_overlap") or 200)
        self.child_chunk_size = int(config.get("child_chunk_size") or 500)
        self.child_chunk_overlap = int(config.get("child_chunk_overlap") or 80)
        self.return_children_only = bool(config.get("return_children_only", True))
        self.return_parent_and_children = bool(config.get("return_parent_and_children", False))
        self.parent_recursive = RecursiveChunker(
            settings,
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
        )
        self.child_recursive = RecursiveChunker(
            settings,
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
        )

    def split(self, document: Document) -> List[ChunkDraft]:
        doc_hash = resolve_doc_hash(
            doc_id=document.id,
            metadata=document.metadata or {},
            text=document.text,
        )
        parents = self.parent_recursive.split(document)
        output: List[ChunkDraft] = []
        child_counter = 0

        for parent_index, parent in enumerate(parents):
            parent_id = generate_chunk_id(
                doc_hash=doc_hash,
                text=parent.text,
                chunk_index=parent_index,
                page_range=parent.page_range,
                section_path=parent.section_path,
                heading_path=parent.heading_path,
                chunk_level="parent",
            )
            parent.chunk_id = parent_id
            parent.chunk_index = parent_index
            parent.metadata["chunk_level"] = "parent"

            child_metadata = {
                **(document.metadata or {}),
                **(parent.metadata or {}),
                "parent_chunk_id": parent_id,
                "chunk_level": "child",
                "heading_path": parent.heading_path,
                "section_path": parent.section_path,
                "page_range": parent.page_range,
            }
            children = self.child_recursive.split_text(
                parent.text,
                base_metadata=child_metadata,
                base_offset=parent.char_start or 0,
                metadata_overrides=child_metadata,
            )
            child_ids: List[str] = []
            for child in children:
                child_id = generate_chunk_id(
                    doc_hash=doc_hash,
                    text=child.text,
                    chunk_index=child_counter,
                    page_range=child.page_range or parent.page_range,
                    section_path=child.section_path or parent.section_path,
                    heading_path=child.heading_path or parent.heading_path,
                    chunk_level="child",
                )
                child.chunk_id = child_id
                child.chunk_index = child_counter
                child.parent_chunk_id = parent_id
                child.metadata["parent_chunk_id"] = parent_id
                child.metadata["chunk_level"] = "child"
                child_ids.append(child_id)
                child_counter += 1

            parent.metadata["child_chunk_ids"] = child_ids

            if self.return_parent_and_children:
                output.append(parent)
                output.extend(children)
            elif self.return_children_only:
                output.extend(children)
            else:
                output.extend(children)

        return output
