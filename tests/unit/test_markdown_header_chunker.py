from types import SimpleNamespace

from src.core.types import Document
from src.ingestion.chunking.markdown_header_chunker import MarkdownHeaderChunker


def _settings(chunk_size=80, preserve_headers=True):
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            chunk_size=chunk_size,
            chunk_overlap=10,
            splitter="recursive",
            chunking=SimpleNamespace(
                strategy="markdown_header",
                chunk_size=chunk_size,
                chunk_overlap=10,
                recursive={"separators": ["\n\n", "\n", " ", ""]},
                markdown_header={
                    "headers": [("#", "h1"), ("##", "h2"), ("###", "h3")],
                    "preserve_headers": preserve_headers,
                    "split_under_headers": True,
                },
                semantic={},
                parent_child={},
                sliding_window={},
            ),
        )
    )


def test_markdown_header_chunker_extracts_heading_path():
    document = Document(
        id="doc1",
        text="# Project Guide\n\n## Installation\n\nInstall dependencies.\n\n## Usage\n\nRun.",
        metadata={"source_path": "README.md"},
    )

    drafts = MarkdownHeaderChunker(_settings()).split(document)

    install = next(d for d in drafts if d.heading == "Installation")
    assert install.heading_path == ["Project Guide", "Installation"]
    assert install.section_path == ["Project Guide", "Installation"]
    assert install.text.startswith("## Installation")


def test_code_block_hash_is_not_heading():
    document = Document(
        id="doc1",
        text="# Guide\n\n```bash\n# not a heading\n```\n\nBody text.",
        metadata={"source_path": "README.md"},
    )

    drafts = MarkdownHeaderChunker(_settings()).split(document)

    assert len(drafts) == 1
    assert drafts[0].heading_path == ["Guide"]


def test_long_heading_content_is_recursively_split():
    long_body = " ".join(["content"] * 80)
    document = Document(
        id="doc1",
        text=f"# Guide\n\n## Long\n\n{long_body}",
        metadata={"source_path": "README.md"},
    )

    drafts = MarkdownHeaderChunker(_settings(chunk_size=60)).split(document)

    assert len([d for d in drafts if d.heading == "Long"]) > 1
    assert all(d.heading_path == ["Guide", "Long"] for d in drafts if d.heading == "Long")


def test_markdown_without_headings_falls_back_to_recursive():
    document = Document(
        id="doc1",
        text="No headings here. " * 10,
        metadata={"source_path": "README.md"},
    )

    drafts = MarkdownHeaderChunker(_settings(chunk_size=60)).split(document)

    assert drafts
    assert all(d.metadata.get("fallback_reason") == "markdown_no_headings" for d in drafts)
