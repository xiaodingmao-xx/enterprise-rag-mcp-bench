"""Unit tests for chunk-level embedding cache."""

from src.core.types import Chunk
from src.ingestion.embedding.embedding_cache import SQLiteEmbeddingCache


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "doc.pdf", "chunk_index": 0},
    )


def test_embedding_cache_reuses_vector_by_content_hash(tmp_path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "embedding_cache.db")
    original = _chunk("a", "same content")
    renamed = _chunk("b", "same content")
    cache.annotate_chunks([original, renamed])

    cache.set_many(
        [original],
        [[0.1, 0.2, 0.3]],
        collection="default",
        provider="openai",
        model="text-embedding-3-small",
        dimensions=3,
    )

    hits = cache.get_many(
        [renamed],
        collection="default",
        provider="openai",
        model="text-embedding-3-small",
        dimensions=3,
    )

    assert hits == {0: [0.1, 0.2, 0.3]}
    assert original.metadata["content_hash"] == renamed.metadata["content_hash"]


def test_embedding_cache_isolated_by_model_and_dimensions(tmp_path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "embedding_cache.db")
    chunk = _chunk("a", "same content")

    cache.set_many(
        [chunk],
        [[0.1, 0.2, 0.3]],
        collection="default",
        provider="openai",
        model="model-a",
        dimensions=3,
    )

    assert cache.get_many(
        [chunk],
        collection="default",
        provider="openai",
        model="model-b",
        dimensions=3,
    ) == {}
    assert cache.get_many(
        [chunk],
        collection="default",
        provider="openai",
        model="model-a",
        dimensions=4,
    ) == {}


def test_disabled_embedding_cache_noops(tmp_path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "embedding_cache.db", enabled=False)
    chunk = _chunk("a", "same content")

    cache.set_many(
        [chunk],
        [[0.1]],
        collection="default",
        provider="openai",
        model="model-a",
        dimensions=1,
    )

    assert cache.get_many(
        [chunk],
        collection="default",
        provider="openai",
        model="model-a",
        dimensions=1,
    ) == {}
