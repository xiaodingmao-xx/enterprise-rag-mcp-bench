"""
Query Engine Module.

This package contains the hybrid search engine components:
- Query preprocessing
- Dense retrieval (embedding-based)
- Sparse retrieval (BM25)
- Result fusion (RRF)
- Reranking
"""

from src.core.query_engine.query_processor import (
    QueryProcessor,
    QueryProcessorConfig,
    create_query_processor,
    DEFAULT_STOPWORDS,
    CHINESE_STOPWORDS,
    ENGLISH_STOPWORDS,
)
from src.core.query_engine.dense_retriever import (
    DenseRetriever,
    create_dense_retriever,
)
from src.core.query_engine.sparse_retriever import (
    SparseRetriever,
    create_sparse_retriever,
)
from src.core.query_engine.fusion import (
    RRFFusion,
    rrf_score,
)
from src.core.query_engine.hybrid_search import (
    HybridSearch,
    HybridSearchConfig,
    HybridSearchResult,
    create_hybrid_search,
)
from src.core.query_engine.retriever import Retriever
from src.core.query_engine.retrieval_filter import RetrievalFilter
from src.core.query_engine.query_rewriter import (
    LLMQueryRewriter,
    NoopQueryRewriter,
    RuleBasedQueryRewriter,
)
from src.core.query_engine.score_booster import ScoreBooster
from src.core.query_engine.tokenizer import DomainTokenizer

__all__ = [
    "QueryProcessor",
    "QueryProcessorConfig",
    "create_query_processor",
    "DEFAULT_STOPWORDS",
    "CHINESE_STOPWORDS",
    "ENGLISH_STOPWORDS",
    "DenseRetriever",
    "create_dense_retriever",
    "SparseRetriever",
    "create_sparse_retriever",
    "RRFFusion",
    "rrf_score",
    "HybridSearch",
    "HybridSearchConfig",
    "HybridSearchResult",
    "create_hybrid_search",
    "Retriever",
    "RetrievalFilter",
    "LLMQueryRewriter",
    "NoopQueryRewriter",
    "RuleBasedQueryRewriter",
    "ScoreBooster",
    "DomainTokenizer",
]
