"""
Vector Store Module.

This package contains vector store abstractions and implementations:
- Base vector store class
- Vector store factory
- Implementations (Chroma, etc.)
"""

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

# Auto-register ChromaStore provider
try:
    from src.libs.vector_store.chroma_store import ChromaStore
    VectorStoreFactory.register_provider('chroma', ChromaStore)
except ImportError:
    # ChromaDB not installed, skip registration
    pass

from src.libs.vector_store.qdrant_store import QdrantStore
from src.libs.vector_store.opensearch_store import OpenSearchStore
from src.libs.vector_store.pgvector_store import PgVectorStore

VectorStoreFactory.register_provider("qdrant", QdrantStore)
VectorStoreFactory.register_provider("opensearch", OpenSearchStore)
VectorStoreFactory.register_provider("pgvector", PgVectorStore)

__all__ = [
    'BaseVectorStore',
    'VectorStoreFactory',
    'ChromaStore',
    'QdrantStore',
    'OpenSearchStore',
    'PgVectorStore',
]
