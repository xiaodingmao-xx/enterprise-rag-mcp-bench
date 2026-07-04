from types import SimpleNamespace

from src.core.types import Document
from src.ingestion.chunking.parent_child_chunker import ParentChildChunker


def _settings(return_parent_and_children=False):
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            chunk_size=80,
            chunk_overlap=10,
            splitter="recursive",
            chunking=SimpleNamespace(
                strategy="parent_child",
                chunk_size=80,
                chunk_overlap=10,
                recursive={"separators": [" ", ""]},
                markdown_header={},
                semantic={},
                parent_child={
                    "parent_chunk_size": 120,
                    "parent_chunk_overlap": 10,
                    "child_chunk_size": 45,
                    "child_chunk_overlap": 5,
                    "return_children_only": not return_parent_and_children,
                    "return_parent_and_children": return_parent_and_children,
                },
                sliding_window={},
            ),
        )
    )


def test_parent_child_children_only_have_parent_id():
    document = Document(
        id="doc1",
        text=" ".join(["alpha beta gamma delta"] * 20),
        metadata={
            "source_path": "doc.txt",
            "heading_path": ["Guide"],
            "page_range": {"start": 2, "end": 2},
        },
    )

    drafts = ParentChildChunker(_settings()).split(document)

    assert drafts
    assert all(d.parent_chunk_id for d in drafts)
    assert all(d.metadata["chunk_level"] == "child" for d in drafts)
    assert all(d.heading_path == ["Guide"] for d in drafts)
    assert all(d.page_range == (2, 2) for d in drafts)


def test_parent_child_can_return_parent_and_children():
    document = Document(
        id="doc1",
        text=" ".join(["alpha beta gamma delta"] * 20),
        metadata={"source_path": "doc.txt"},
    )

    drafts = ParentChildChunker(_settings(return_parent_and_children=True)).split(document)

    parents = [d for d in drafts if d.metadata.get("chunk_level") == "parent"]
    children = [d for d in drafts if d.metadata.get("chunk_level") == "child"]
    assert parents
    assert children
    assert parents[0].metadata["child_chunk_ids"]
    assert children[0].chunk_id in parents[0].metadata["child_chunk_ids"]
    assert children[0].parent_chunk_id == parents[0].chunk_id
