from types import SimpleNamespace

from src.core.types import Document
from src.ingestion.chunking.document_chunker import DocumentChunker


def _settings(strategy="markdown_header"):
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            chunk_size=80,
            chunk_overlap=10,
            splitter="recursive",
            chunking=SimpleNamespace(
                strategy=strategy,
                chunk_size=80,
                chunk_overlap=10,
                recursive={"separators": ["\n\n", "\n", " ", ""]},
                markdown_header={
                    "headers": [("#", "h1"), ("##", "h2")],
                    "preserve_headers": True,
                    "split_under_headers": True,
                },
                semantic={},
                parent_child={},
                sliding_window={},
            ),
        )
    )


def test_document_chunker_outputs_structured_metadata_and_refs():
    document = Document(
        id="doc1",
        text="# Guide\n\n## Install\n\nUse [IMAGE: img1] and [TABLE: tbl1].",
        metadata={
            "source_path": "guide.md",
            "doc_hash": "hash1",
            "images": [{"id": "img1", "path": "img.png", "page": 1}],
            "table_ids": ["tbl1"],
        },
    )

    chunks = DocumentChunker(_settings()).split_document(document)
    install = next(c for c in chunks if c.metadata.get("heading") == "Install")

    assert install.id == install.metadata["chunk_id"]
    assert install.metadata["doc_id"] == "doc1"
    assert install.metadata["source_ref"] == "doc1"
    assert isinstance(install.metadata["chunk_index"], int)
    assert install.metadata["heading_path"] == ["Guide", "Install"]
    assert install.metadata["section_path"] == ["Guide", "Install"]
    assert install.metadata["heading_path_text"] == "Guide > Install"
    assert install.metadata["section_path_text"] == "Guide > Install"
    assert install.metadata["image_ids"] == ["img1"]
    assert install.metadata["image_ids_text"] == "img1"
    assert install.metadata["table_ids"] == ["tbl1"]
    assert install.metadata["table_ids_text"] == "tbl1"
    assert install.metadata["images"][0]["id"] == "img1"
    assert install.metadata["text"] == install.text
