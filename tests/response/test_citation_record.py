from src.core.response.citation import CitationRecord
from src.core.response.retrieval_status import contexts_from_results
from src.core.types import RetrievalResult


def test_citation_record_reads_document_version_and_page_metadata() -> None:
    result = RetrievalResult(
        chunk_id="chunk-1",
        score=0.87,
        text="Evidence text",
        metadata={
            "source_path": "docs/policy.md",
            "document_id": "doc-1",
            "version_id": "v2",
            "page_start": 3,
            "page_end": 4,
        },
    )
    context = contexts_from_results([result])[0]
    record = CitationRecord.from_context(context)

    payload = record.to_dict()
    assert payload["citation_id"] == "C1"
    assert payload["document_id"] == "doc-1"
    assert payload["version_id"] == "v2"
    assert payload["page_start"] == 3
    assert payload["source"] == "docs/policy.md"

