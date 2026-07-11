import pytest

from src.core.query_engine.query_rewriter import LLMQueryRewriter, RuleBasedQueryRewriter
from src.core.query_engine.retrieval_filter import InvalidRetrievalFilterError, RetrievalFilter
from src.core.query_engine.score_booster import ScoreBoostConfig, ScoreBooster
from src.core.query_engine.tokenizer import DomainTokenizer, TokenizerConfig
from src.core.types import RetrievalResult
from src.security.context import RequestContext


def test_retrieval_filter_validates_and_merges_request_identity():
    retrieval_filter = RetrievalFilter.from_dict(
        {
            "tenant_id": "tenant-a",
            "document_ids": ["doc-1"],
            "source_types": ["pdf"],
            "tags": ["finance"],
        }
    )
    assert retrieval_filter.matches(
        {
            "tenant_id": "tenant-a",
            "document_id": "doc-1",
            "source_type": "pdf",
            "tags": ["finance", "internal"],
        }
    )
    assert not retrieval_filter.matches({"tenant_id": "tenant-b", "document_id": "doc-1"})
    context = RequestContext(tenant_id="tenant-a", user_id="u1", roles=("reader",), department="legal")
    merged = retrieval_filter.merge_with_request_context(context)
    assert merged.to_dict()["acl"]["user_id"] == "u1"
    assert merged.department == "legal"
    assert merged.to_metadata_filter()["tenant_id"] == "tenant-a"

    with pytest.raises(InvalidRetrievalFilterError):
        RetrievalFilter.from_dict({"unknown": "value"})


def test_domain_tokenizer_preserves_production_terms():
    tokenizer = DomainTokenizer(TokenizerConfig())
    tokens = tokenizer.tokenize("gpt-4 machine-learning Qwen3.7 text-embedding-v4")
    assert tokens == ["gpt-4", "machine-learning", "Qwen3.7", "text-embedding-v4"]


def test_query_rewriter_rule_based_and_llm_fallback():
    rule_based = RuleBasedQueryRewriter(synonyms={"报销": ["费用报销"]}, abbreviations={"RAG": "retrieval augmented generation"})
    result = rule_based.rewrite("RAG 报销")
    assert result.rewrite_strategy == "rule_based"
    assert result.generated_queries

    rewriter = LLMQueryRewriter(rewrite_fn=lambda _query: (_ for _ in ()).throw(RuntimeError("offline")))
    fallback = rewriter.rewrite("query")
    assert fallback.fallback_used is True
    assert fallback.rewrite_strategy == "noop"


def test_score_booster_records_reasons_and_reranks():
    results = [
        RetrievalResult("c1", 0.5, "how to configure", {"title": "Configure Azure", "source_type": "pdf"}),
        RetrievalResult("c2", 0.8, "other", {"title": "Other", "source_type": "md"}),
    ]
    booster = ScoreBooster(
        ScoreBoostConfig(enabled=True, title_boost=2.0, exact_phrase_boost=1.0)
    )
    boosted = booster.apply("configure", results)
    assert boosted[0].chunk_id == "c1"
    assert "title" in boosted[0].metadata["boost_reasons"]
