from src.core.response.citation_verifier import CitationVerifier
from src.core.response.claim_extractor import RuleBasedClaimExtractor
from src.core.response.confidence import AnswerConfidenceScorer
from src.core.response.retrieval_status import RetrievalStatus, contexts_from_results
from src.core.types import RetrievalResult


def test_confidence_uses_citation_and_support_factors() -> None:
    result = RetrievalResult(
        chunk_id="c1",
        score=0.9,
        text="The feature is enabled.",
        metadata={"source_path": "a.md"},
    )
    contexts = contexts_from_results([result])
    answer = "The feature is enabled. [C1]"
    verification = CitationVerifier().verify(
        answer=answer,
        claims=RuleBasedClaimExtractor().extract(answer),
        contexts=contexts,
    )
    confidence = AnswerConfidenceScorer().score(
        answer=answer,
        results=[result],
        contexts=contexts,
        retrieval_status=RetrievalStatus.SUFFICIENT,
        citation_result=verification,
    )

    assert 0 <= confidence.score <= 1
    assert confidence.factors["citation_coverage"] == 1.0
    assert "top_retrieval_score" in confidence.factors

