from src.core.response.citation_verifier import CitationVerifier
from src.core.response.claim_extractor import RuleBasedClaimExtractor
from src.core.response.retrieval_status import contexts_from_results
from src.security.context import RequestContext
from src.core.types import RetrievalResult


def _context(text: str, tenant_id: str = "tenant_a"):
    return contexts_from_results([
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.9,
            text=text,
            metadata={"source_path": "docs/a.md", "tenant_id": tenant_id, "page": 1},
        )
    ])


def test_invalid_marker_is_reported() -> None:
    contexts = _context("The feature is enabled.")
    answer = "The feature is enabled. [C99]"
    result = CitationVerifier().verify(
        answer=answer,
        claims=RuleBasedClaimExtractor().extract(answer),
        contexts=contexts,
    )

    assert result.status.value == "invalid_citation"
    assert "C99" in result.invalid_citations
    assert "INVALID_CITATION" in result.warnings


def test_unauthorized_citation_is_removed() -> None:
    contexts = _context("The feature is enabled.", tenant_id="tenant_b")
    answer = "The feature is enabled. [C1]"
    request_context = RequestContext(
        tenant_id="tenant_a", user_id="user-a", authenticated=True
    )
    result = CitationVerifier().verify(
        answer=answer,
        claims=RuleBasedClaimExtractor().extract(answer),
        contexts=contexts,
        request_context=request_context,
    )

    assert result.valid_citations == []
    assert "UNAUTHORIZED_CITATION_REMOVED" in result.warnings
