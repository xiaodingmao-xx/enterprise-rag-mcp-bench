import time

from src.core.query_engine.reranker import CoreReranker, RerankConfig
from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker


class SlowFakeReranker(BaseReranker):
    def rerank(self, query, candidates, trace=None, **kwargs):
        time.sleep(0.05)
        return candidates


class Settings:
    class Rerank:
        enabled = True
        provider = "fake"
        model = "fake-model"
        top_k = 1

    rerank = Rerank()


def test_timeout_fallback_records_structured_error_code() -> None:
    reranker = CoreReranker(
        settings=Settings(),
        reranker=SlowFakeReranker(),
        config=RerankConfig(
            enabled=True,
            top_k=1,
            candidate_top_k=2,
            output_top_k=1,
            timeout_seconds=0.001,
            fallback_on_timeout=True,
        ),
    )
    results = [
        RetrievalResult(chunk_id="a", score=0.9, text="a", metadata={}),
        RetrievalResult(chunk_id="b", score=0.8, text="b", metadata={}),
    ]

    result = reranker.rerank("q", results)

    assert result.fallback_used is True
    assert result.timeout is True
    assert result.error_code == "RERANK_TIMEOUT"
    assert [item.chunk_id for item in result.results] == ["a"]

