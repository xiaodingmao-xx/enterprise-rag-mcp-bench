from types import SimpleNamespace

import pytest

from src.libs.vector_store.opensearch_store import OpenSearchStore
from src.libs.vector_store.pgvector_store import PgVectorStore
from src.libs.vector_store.qdrant_store import QdrantStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


@pytest.mark.parametrize(
    ("provider", "provider_class", "block", "error"),
    [
        ("qdrant", QdrantStore, {"url": "http://localhost:6333"}, "Qdrant"),
        ("opensearch", OpenSearchStore, {"hosts": ["http://localhost:9200"]}, "OpenSearch"),
        ("pgvector", PgVectorStore, {"dsn": "postgresql://localhost/rag"}, "PgVector"),
    ],
)
def test_reserved_vector_backends_are_configurable_but_explicitly_incomplete(provider, provider_class, block, error):
    settings = SimpleNamespace(vector_store=SimpleNamespace(**{provider: block}))
    store = provider_class(settings=settings)
    with pytest.raises(NotImplementedError, match="reserved"):
        store.query([0.1], top_k=1)
    assert provider in VectorStoreFactory.list_providers()


@pytest.mark.parametrize(
    ("provider_class", "provider", "message"),
    [
        (QdrantStore, "qdrant", "url"),
        (OpenSearchStore, "opensearch", "hosts"),
        (PgVectorStore, "pgvector", "dsn"),
    ],
)
def test_reserved_vector_backends_fail_fast_without_connection_config(provider_class, provider, message):
    settings = SimpleNamespace(vector_store=SimpleNamespace(**{provider: {}}))
    with pytest.raises(ValueError, match=message):
        provider_class(settings=settings)
