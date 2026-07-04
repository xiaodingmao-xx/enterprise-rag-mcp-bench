from src.libs.vector_store.chroma_store import ChromaStore


def test_chroma_metadata_sanitizes_structured_chunk_fields():
    store = object.__new__(ChromaStore)
    metadata = {
        "source_path": "guide.md",
        "doc_id": "doc1",
        "chunk_id": "hash::c0000::abcd1234",
        "chunk_index": 0,
        "parent_chunk_id": "parent1",
        "heading_path": ["Guide", "Install"],
        "heading_path_text": "Guide > Install",
        "page_range": {"start": 2, "end": 3},
        "page_range_text": "2-3",
    }

    sanitized = store._sanitize_metadata(metadata)

    assert sanitized["source_path"] == "guide.md"
    assert sanitized["doc_id"] == "doc1"
    assert sanitized["chunk_id"] == "hash::c0000::abcd1234"
    assert sanitized["chunk_index"] == 0
    assert sanitized["parent_chunk_id"] == "parent1"
    assert sanitized["heading_path"] == "Guide,Install"
    assert sanitized["heading_path_text"] == "Guide > Install"
    assert sanitized["page_range_text"] == "2-3"
    assert isinstance(sanitized["page_range"], str)
