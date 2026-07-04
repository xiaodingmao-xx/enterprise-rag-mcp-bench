import pytest

from src.core.types import Document
from src.ingestion.chunking.chunk_schema import ChunkDraft, draft_to_chunk


def test_chunk_schema_syncs_metadata_fields():
    document = Document(
        id="doc1",
        text="# Guide\n\nInstall deps [IMAGE: img1]",
        metadata={
            "source_path": "guide.md",
            "doc_hash": "hash1",
            "images": [{"id": "img1", "path": "img.png", "page": 2}],
        },
    )
    draft = ChunkDraft(
        text="## Install\n\nInstall deps [IMAGE: img1]",
        heading_path=["Guide", "Install"],
        section_path=["Guide", "Install"],
        heading="Install",
        page_range=(2, 3),
        char_start=10,
        char_end=45,
    )

    chunk = draft_to_chunk(
        document=document,
        draft=draft,
        chunk_id="hash1::p002::secguide-install::c0000",
        chunk_index=0,
    )

    assert chunk.metadata["chunk_id"] == chunk.id
    assert chunk.metadata["doc_id"] == "doc1"
    assert chunk.metadata["page_range"] == {"start": 2, "end": 3}
    assert chunk.metadata["page_range_text"] == "2-3"
    assert chunk.metadata["heading_path_text"] == "Guide > Install"
    assert chunk.metadata["section_path_text"] == "Guide > Install"
    assert chunk.metadata["image_ids"] == ["img1"]
    assert chunk.metadata["image_ids_text"] == "img1"
    assert chunk.metadata["images"][0]["id"] == "img1"
    assert chunk.char_start == 10
    assert chunk.char_end == 45


def test_empty_chunk_text_rejected():
    with pytest.raises(ValueError, match="Chunk text cannot be empty"):
        ChunkDraft(text="   ")
