"""Unit tests for CoreReranker fallback behavior.

This module tests the CoreReranker class with focus on:
1. Normal reranking flow with mocked backend
2. Fallback behavior when backend fails
3. Configuration handling
4. Type conversion between RetrievalResult and reranker format
"""

import time

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from src.core.query_engine.reranker import (
    CoreReranker,
    RerankConfig,
    RerankError,
    RerankResult,
    create_core_reranker,
)
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker


# =============================================================================
# Fixtures
# =============================================================================

class MockSettings:
    """Mock Settings class for testing."""
    
    def __init__(
        self,
        enabled: bool = True,
        provider: str = "llm",
        top_k: int = 5,
    ):
        self.rerank = MagicMock()
        self.rerank.enabled = enabled
        self.rerank.provider = provider
        self.rerank.top_k = top_k


class FakeReranker(BaseReranker):
    """Fake reranker that returns candidates in reverse order."""
    
    def __init__(self, settings: Any = None, **kwargs: Any):
        self.call_count = 0
        self.last_query = None
        self.last_candidates = None
        self.last_kwargs = None
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.call_count += 1
        self.last_query = query
        self.last_candidates = candidates
        self.last_kwargs = kwargs
        
        # Reverse order and add rerank_score
        reranked = []
        for i, candidate in enumerate(reversed(candidates)):
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = 3.0 - i * 0.5  # Descending scores
            reranked.append(candidate_copy)
        
        return reranked


class FailingReranker(BaseReranker):
    """Reranker that always fails."""
    
    def __init__(self, settings: Any = None, error_message: str = "Reranker failed", **kwargs: Any):
        self.error_message = error_message
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        raise RuntimeError(self.error_message)


class SlowReranker(BaseReranker):
    """Reranker that blocks longer than the configured timeout."""

    def __init__(self, sleep_seconds: float = 0.2):
        self.sleep_seconds = sleep_seconds
        self.call_count = 0

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.call_count += 1
        time.sleep(self.sleep_seconds)
        return candidates


class InvalidOutputReranker(BaseReranker):
    """Reranker that returns malformed output."""

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        return []


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return MockSettings(enabled=True, provider="llm", top_k=5)


@pytest.fixture
def sample_results():
    """Create sample RetrievalResults for testing."""
    return [
        RetrievalResult(
            chunk_id="chunk_001",
            score=0.9,
            text="First chunk about machine learning.",
            metadata={"source_path": "doc1.pdf", "page": 1},
        ),
        RetrievalResult(
            chunk_id="chunk_002",
            score=0.8,
            text="Second chunk about deep learning.",
            metadata={"source_path": "doc1.pdf", "page": 2},
        ),
        RetrievalResult(
            chunk_id="chunk_003",
            score=0.7,
            text="Third chunk about neural networks.",
            metadata={"source_path": "doc2.pdf", "page": 1},
        ),
    ]


# =============================================================================
# Test: Initialization
# =============================================================================

class TestCoreRerankerInit:
    """Test CoreReranker initialization."""
    
    def test_init_with_custom_reranker(self, mock_settings):
        """Test initialization with custom reranker backend."""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        assert reranker._reranker is fake_reranker
        assert reranker.reranker_type == "fakereranker"
    
    def test_init_with_disabled_config(self):
        """Test initialization with disabled reranking."""
        settings = MockSettings(enabled=False, provider="none")
        reranker = CoreReranker(settings=settings)
        
        assert isinstance(reranker._reranker, NoneReranker)
        assert reranker.reranker_type == "none"
        assert not reranker.is_enabled
    
    def test_init_with_custom_config(self, mock_settings):
        """Test initialization with custom RerankConfig."""
        config = RerankConfig(enabled=True, top_k=10, timeout=60.0)
        fake_reranker = FakeReranker()
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=fake_reranker,
            config=config,
        )
        
        assert reranker.config.top_k == 10
        assert reranker.config.timeout == 60.0
    
    def test_init_extracts_config_from_settings(self, mock_settings):
        """Test that config is extracted from settings when not provided."""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        assert reranker.config.enabled is True
        assert reranker.config.top_k == 5


# =============================================================================
# Test: Normal Reranking Flow
# =============================================================================

class TestCoreRerankerNormalFlow:
    """Test normal reranking flow."""
    
    def test_rerank_calls_backend(self, mock_settings, sample_results):
        """Test that reranking calls the backend reranker."""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        reranker.rerank("test query", sample_results)
        
        assert fake_reranker.call_count == 1
        assert fake_reranker.last_query == "test query"
        assert len(fake_reranker.last_candidates) == 3
    
    def test_rerank_returns_reranked_results(self, mock_settings, sample_results):
        """Test that reranked results are returned correctly."""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        assert isinstance(result, RerankResult)
        assert len(result.results) == 3
        assert result.used_fallback is False
        assert result.reranker_type == "fakereranker"
    
    def test_rerank_preserves_original_data(self, mock_settings, sample_results):
        """Test that original result data is preserved."""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        # FakeReranker reverses order, so first result should be chunk_003
        first = result.results[0]
        assert first.chunk_id == "chunk_003"
        assert first.text == "Third chunk about neural networks."
        assert first.metadata["source_path"] == "doc2.pdf"
    
    def test_rerank_adds_score_metadata(self, mock_settings, sample_results):
        """Test that rerank scores are added to metadata."""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        first = result.results[0]
        assert "original_score" in first.metadata
        assert "rerank_score" in first.metadata
        assert first.metadata["reranked"] is True
    
    def test_rerank_respects_top_k(self, mock_settings, sample_results):
        """Test that top_k limits results."""
        fake_reranker = FakeReranker()
        config = RerankConfig(enabled=True, top_k=2)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=fake_reranker,
            config=config,
        )
        
        result = reranker.rerank("test query", sample_results)
        
        assert len(result.results) == 2
    
    def test_rerank_explicit_top_k_overrides_config(self, mock_settings, sample_results):
        """Test that explicit top_k parameter overrides config."""
        fake_reranker = FakeReranker()
        config = RerankConfig(enabled=True, top_k=5)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=fake_reranker,
            config=config,
        )
        
        result = reranker.rerank("test query", sample_results, top_k=1)
        
        assert len(result.results) == 1

    def test_rerank_limits_candidates_and_output(self, mock_settings, sample_results):
        """Test candidate_top_k limits input and output_top_k limits output."""
        fake_reranker = FakeReranker()
        config = RerankConfig(
            enabled=True,
            top_k=5,
            candidate_top_k=2,
            output_top_k=1,
        )
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=fake_reranker,
            config=config,
        )

        result = reranker.rerank("test query", sample_results)

        assert len(fake_reranker.last_candidates) == 2
        assert [c["id"] for c in fake_reranker.last_candidates] == [
            "chunk_001",
            "chunk_002",
        ]
        assert len(result.results) == 1
        assert result.results[0].chunk_id == "chunk_002"

    def test_rerank_does_not_inject_provider_top_k_kwarg(
        self,
        mock_settings,
        sample_results,
    ):
        """Test CoreReranker keeps provider kwargs compatible with LLM rerank."""
        fake_reranker = FakeReranker()
        config = RerankConfig(enabled=True, top_k=5, output_top_k=1)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=fake_reranker,
            config=config,
        )

        reranker.rerank("test query", sample_results, top_k=1)

        assert fake_reranker.last_kwargs == {}


# =============================================================================
# Test: Fallback Behavior
# =============================================================================

class TestCoreRerankerFallback:
    """Test fallback behavior when backend fails."""
    
    def test_fallback_on_backend_error(self, mock_settings, sample_results):
        """Test that fallback returns original order on error."""
        failing_reranker = FailingReranker(error_message="Backend crashed")
        reranker = CoreReranker(settings=mock_settings, reranker=failing_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        assert result.used_fallback is True
        assert "Backend crashed" in result.fallback_reason
        assert len(result.results) == 3
    
    def test_fallback_preserves_original_order(self, mock_settings, sample_results):
        """Test that fallback preserves original order."""
        failing_reranker = FailingReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=failing_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        assert result.results[0].chunk_id == "chunk_001"
        assert result.results[1].chunk_id == "chunk_002"
        assert result.results[2].chunk_id == "chunk_003"
    
    def test_fallback_marks_results(self, mock_settings, sample_results):
        """Test that fallback results are marked appropriately."""
        failing_reranker = FailingReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=failing_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        for r in result.results:
            assert r.metadata.get("reranked") is False
            assert r.metadata.get("rerank_fallback") is True
    
    def test_fallback_respects_top_k(self, mock_settings, sample_results):
        """Test that fallback respects top_k."""
        failing_reranker = FailingReranker()
        config = RerankConfig(enabled=True, top_k=2, fallback_on_error=True)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=failing_reranker,
            config=config,
        )
        
        result = reranker.rerank("test query", sample_results)
        
        assert len(result.results) == 2
    
    def test_fallback_disabled_raises_error(self, mock_settings, sample_results):
        """Test that error is raised when fallback is disabled."""
        failing_reranker = FailingReranker()
        config = RerankConfig(enabled=True, top_k=5, fallback_on_error=False)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=failing_reranker,
            config=config,
        )
        
        with pytest.raises(RerankError) as exc_info:
            reranker.rerank("test query", sample_results)
        
        assert "fallback disabled" in str(exc_info.value)
    
    def test_stores_original_order_on_fallback(self, mock_settings, sample_results):
        """Test that original order is stored in result."""
        failing_reranker = FailingReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=failing_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        assert result.original_order is not None
        assert len(result.original_order) == 3

    def test_timeout_fallback_returns_original_results_and_trace(
        self,
        mock_settings,
        sample_results,
    ):
        """Test timeout uses original hybrid-search order and records trace metadata."""
        slow_reranker = SlowReranker(sleep_seconds=0.2)
        config = RerankConfig(
            enabled=True,
            top_k=2,
            candidate_top_k=3,
            output_top_k=2,
            timeout_seconds=0.01,
            fallback_on_timeout=True,
        )
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=slow_reranker,
            config=config,
        )
        trace = TraceContext(trace_type="query")

        started = time.monotonic()
        result = reranker.rerank("test query", sample_results, trace=trace)
        elapsed = time.monotonic() - started

        assert elapsed < 0.15
        assert result.used_fallback is True
        assert result.fallback_reason is not None
        assert "timeout" in result.fallback_reason
        assert [r.chunk_id for r in result.results] == ["chunk_001", "chunk_002"]
        assert trace.metadata["fallback_reason"] == "timeout"
        assert trace.metadata["timeout_seconds"] == 0.01
        assert trace.metadata["candidate_count"] == 3
        rerank_stage = trace.get_stage_data("rerank")
        assert rerank_stage is not None
        assert rerank_stage["used_fallback"] is True
        assert rerank_stage["fallback_reason"] == "timeout"

    def test_exception_fallback_records_reason_and_does_not_raise(
        self,
        mock_settings,
        sample_results,
    ):
        """Test backend exception falls back without failing the query path."""
        failing_reranker = FailingReranker(error_message="Backend crashed")
        config = RerankConfig(
            enabled=True,
            top_k=2,
            candidate_top_k=3,
            output_top_k=2,
            fallback_on_timeout=True,
        )
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=failing_reranker,
            config=config,
        )
        trace = TraceContext(trace_type="query")

        result = reranker.rerank("test query", sample_results, trace=trace)

        assert result.used_fallback is True
        assert "Backend crashed" in result.fallback_reason
        assert [r.chunk_id for r in result.results] == ["chunk_001", "chunk_002"]
        assert trace.metadata["fallback_reason"] == "rerank_error"
        assert trace.metadata["candidate_count"] == 3

    def test_invalid_output_fallback_returns_original_top_k(
        self,
        mock_settings,
        sample_results,
    ):
        """Test abnormal rerank output falls back to original Hybrid Search top-k."""
        invalid_reranker = InvalidOutputReranker()
        config = RerankConfig(
            enabled=True,
            top_k=2,
            candidate_top_k=3,
            output_top_k=2,
            fallback_on_timeout=True,
        )
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=invalid_reranker,
            config=config,
        )

        result = reranker.rerank("test query", sample_results)

        assert result.used_fallback is True
        assert result.fallback_reason is not None
        assert "invalid_rerank_output" in result.fallback_reason
        assert [r.chunk_id for r in result.results] == ["chunk_001", "chunk_002"]
        for item in result.results:
            assert item.metadata["rerank_fallback"] is True
            assert item.metadata["fallback_reason"] == "invalid_rerank_output"


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestCoreRerankerEdgeCases:
    """Test edge cases."""
    
    def test_empty_results(self, mock_settings):
        """Test handling of empty results."""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        result = reranker.rerank("test query", [])
        
        assert result.results == []
        assert result.used_fallback is False
        assert fake_reranker.call_count == 0  # Should not call backend
    
    def test_single_result(self, mock_settings):
        """Test handling of single result."""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        single_result = [RetrievalResult(
            chunk_id="only_one",
            score=0.9,
            text="Single result",
            metadata={"source_path": "test.pdf"},
        )]
        
        result = reranker.rerank("test query", single_result)
        
        assert len(result.results) == 1
        assert result.results[0].chunk_id == "only_one"
        assert fake_reranker.call_count == 0  # Should not call backend for single result
    
    def test_disabled_reranking(self, sample_results):
        """Test that disabled reranking returns original order."""
        settings = MockSettings(enabled=False, provider="none")
        reranker = CoreReranker(settings=settings)
        
        result = reranker.rerank("test query", sample_results)
        
        assert result.reranker_type == "none"
        assert result.results[0].chunk_id == "chunk_001"
        assert result.used_fallback is False
    
    def test_none_reranker_returns_top_k(self, sample_results):
        """Test that NoneReranker returns top_k results."""
        settings = MockSettings(enabled=False, provider="none", top_k=2)
        reranker = CoreReranker(settings=settings)
        
        result = reranker.rerank("test query", sample_results)
        
        assert len(result.results) == 2


# =============================================================================
# Test: Type Conversion
# =============================================================================

class TestTypeConversion:
    """Test type conversion between RetrievalResult and reranker format."""
    
    def test_results_to_candidates(self, mock_settings, sample_results):
        """Test conversion from RetrievalResult to candidate dict."""
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=FakeReranker(),
        )
        
        candidates = reranker._results_to_candidates(sample_results)
        
        assert len(candidates) == 3
        assert candidates[0]["id"] == "chunk_001"
        assert candidates[0]["text"] == "First chunk about machine learning."
        assert candidates[0]["score"] == 0.9
        assert "source_path" in candidates[0]["metadata"]
    
    def test_candidates_to_results(self, mock_settings, sample_results):
        """Test conversion from candidate dict back to RetrievalResult."""
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=FakeReranker(),
        )
        
        candidates = [
            {"id": "chunk_001", "text": "Updated text", "rerank_score": 2.5},
            {"id": "chunk_002", "text": "Another text", "rerank_score": 2.0},
        ]
        
        results = reranker._candidates_to_results(candidates, sample_results)
        
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_001"
        assert results[0].score == 2.5
        assert results[0].metadata.get("original_score") == 0.9


# =============================================================================
# Test: Factory Function
# =============================================================================

class TestCreateCoreReranker:
    """Test create_core_reranker factory function."""
    
    def test_creates_reranker(self, mock_settings):
        """Test that factory creates CoreReranker."""
        fake_reranker = FakeReranker()
        reranker = create_core_reranker(mock_settings, reranker=fake_reranker)
        
        assert isinstance(reranker, CoreReranker)
        assert reranker._reranker is fake_reranker


# =============================================================================
# Test: Properties
# =============================================================================

class TestCoreRerankerProperties:
    """Test CoreReranker properties."""
    
    def test_reranker_type_property(self, mock_settings):
        """Test reranker_type property."""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        assert reranker.reranker_type == "fakereranker"
    
    def test_is_enabled_true(self, mock_settings):
        """Test is_enabled returns True when enabled."""
        fake_reranker = FakeReranker()
        config = RerankConfig(enabled=True)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=fake_reranker,
            config=config,
        )
        
        assert reranker.is_enabled is True
    
    def test_is_enabled_false_when_disabled(self):
        """Test is_enabled returns False when disabled."""
        settings = MockSettings(enabled=False)
        reranker = CoreReranker(settings=settings)
        
        assert reranker.is_enabled is False
    
    def test_is_enabled_false_for_none_reranker(self, mock_settings):
        """Test is_enabled returns False for NoneReranker."""
        none_reranker = NoneReranker(settings=mock_settings)
        config = RerankConfig(enabled=True)  # Config says enabled
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=none_reranker,  # But using NoneReranker
            config=config,
        )
        
        assert reranker.is_enabled is False
