from __future__ import annotations

from src.core.types import Document
from src.libs.loader.parsed_document import Block, ExtractionQuality, Image, Page, ParsedDocument, Table


def test_parsed_document_serializes_and_adapts_without_full_chunk_payload():
    block = Block("b1", "paragraph", "hello", 1, (0.0, 0.0, 10.0, 10.0))
    table = Table(
        table_id="t1",
        page_number=1,
        headers=["A"],
        rows=[["1"]],
        markdown="| A |\n| --- |\n| 1 |",
        plain_text="A\n1",
    )
    image = Image("i1", 1, bbox=(1.0, 2.0, 3.0, 4.0), extracted_path="image.png")
    parsed = ParsedDocument(
        document_id="doc1",
        source_path="sample.pdf",
        source_type="pdf",
        pages=[Page(1, 100, 100, "hello", 5, [block])],
        blocks=[block],
        paragraphs=[block],
        tables=[table],
        images=[image],
        extraction_quality=ExtractionQuality(text_density=5, quality_status="warning"),
        text="hello",
    )

    payload = parsed.to_dict()
    assert payload["tables"][0]["table_id"] == "t1"
    assert '"document_id": "doc1"' in parsed.to_json()
    document = parsed.to_document()
    assert isinstance(document, Document)
    assert document.metadata["table_refs"][0]["table_id"] == "t1"
    assert document.metadata["image_refs"][0]["image_id"] == "i1"
    assert "parsed_document_summary" in document.metadata
