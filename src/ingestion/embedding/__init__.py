"""
Embedding Module.

This package contains embedding components:
- Dense encoder
- Sparse encoder (BM25)
- Batch processor
"""

from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor, BatchResult
from src.ingestion.embedding.embedding_cache import SQLiteEmbeddingCache

__all__ = [
    "DenseEncoder",
    "SparseEncoder",
    "BatchProcessor",
    "BatchResult",
    "SQLiteEmbeddingCache",
]
