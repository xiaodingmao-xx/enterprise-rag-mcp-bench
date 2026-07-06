"""Tests for SQLite FTS5 sparse indexer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.core.types import Chunk
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.ingestion.storage.sqlite_fts5_indexer import SQLiteFTS5Indexer


def _chunk(chunk_id: str, text: str, source_path: str = "doc.txt") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": source_path, "chunk_index": 0},
    )


def _indexer(tmp_path: Path) -> SQLiteFTS5Indexer:
    return SQLiteFTS5Indexer(db_path=str(tmp_path / "sparse_fts5.db"))


def test_fts5_add_and_query(tmp_path: Path) -> None:
    indexer = _indexer(tmp_path)
    chunks = [
        _chunk("c1", "Azure OpenAI embeddings and Chroma retrieval.", "a.txt"),
        _chunk("c2", "Lunch plans and office tea rotation.", "b.txt"),
    ]

    indexer.add_documents(
        [{"doc_length": 5}, {"doc_length": 6}],
        chunks=chunks,
        collection="docs",
        doc_id="doc-a",
        source_path="a.txt",
    )

    results = indexer.query(["embeddings"], collection="docs", top_k=5)

    assert [result["chunk_id"] for result in results] == ["c1"]
    assert results[0]["metadata"]["source_path"] == "a.txt"


def test_fts5_reingest_replaces_old_chunks(tmp_path: Path) -> None:
    indexer = _indexer(tmp_path)
    indexer.add_documents(
        [{"doc_length": 3}],
        chunks=[_chunk("old", "obsolete cache notes", "doc.txt")],
        collection="docs",
        doc_id="doc-1",
        source_path="doc.txt",
    )

    indexer.add_documents(
        [{"doc_length": 4}],
        chunks=[_chunk("new", "fresh cache notes", "doc.txt")],
        collection="docs",
        doc_id="doc-1",
        source_path="doc.txt",
    )

    assert indexer.query(["obsolete"], collection="docs") == []
    assert [r["chunk_id"] for r in indexer.query(["fresh"], collection="docs")] == ["new"]
    assert indexer.count_chunks("docs") == 1


def test_fts5_remove_document(tmp_path: Path) -> None:
    indexer = _indexer(tmp_path)
    indexer.add_documents(
        [{"doc_length": 3}],
        chunks=[_chunk("c1", "delete me from sparse index", "delete.txt")],
        collection="docs",
        doc_id="doc-delete",
        source_path="delete.txt",
    )

    assert indexer.remove_document("doc-delete", "docs") is True

    assert indexer.query(["delete"], collection="docs") == []


def test_fts5_collection_isolation(tmp_path: Path) -> None:
    indexer = _indexer(tmp_path)
    indexer.add_documents(
        [{"doc_length": 2}],
        chunks=[_chunk("a", "shared keyword", "a.txt")],
        collection="a",
        doc_id="doc-a",
        source_path="a.txt",
    )
    indexer.add_documents(
        [{"doc_length": 2}],
        chunks=[_chunk("b", "shared keyword", "b.txt")],
        collection="b",
        doc_id="doc-b",
        source_path="b.txt",
    )

    assert [r["chunk_id"] for r in indexer.query(["shared"], collection="a")] == ["a"]
    assert [r["chunk_id"] for r in indexer.query(["shared"], collection="b")] == ["b"]


def test_fts5_special_chars_query(tmp_path: Path) -> None:
    indexer = _indexer(tmp_path)
    indexer.add_documents(
        [{"doc_length": 8}],
        chunks=[
            _chunk(
                "c1",
                "Use text-embedding-v4 with data/db/chroma and query_knowledge_hub.",
                "special.txt",
            )
        ],
        collection="docs",
        doc_id="doc-special",
        source_path="special.txt",
    )

    assert indexer.query(["text-embedding-v4"], collection="docs")
    assert indexer.query(["data/db/chroma"], collection="docs")
    assert indexer.query(["query_knowledge_hub"], collection="docs")


def test_fts5_concurrent_writes(tmp_path: Path) -> None:
    indexer = _indexer(tmp_path)

    def write_doc(i: int) -> None:
        indexer.add_documents(
            [{"doc_length": 4}],
            chunks=[_chunk(f"c{i}", f"concurrent write token {i}", f"{i}.txt")],
            collection="docs",
            doc_id=f"doc-{i}",
            source_path=f"{i}.txt",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write_doc, range(12)))

    assert indexer.count_chunks("docs") == 12
    assert len(indexer.query(["concurrent"], collection="docs", top_k=20)) == 12


def test_sparse_retriever_with_fts5_indexer(tmp_path: Path) -> None:
    indexer = _indexer(tmp_path)
    chunk = _chunk("c1", "retriever bridge uses fts5 sparse backend", "bridge.txt")
    indexer.add_documents(
        [{"doc_length": 6}],
        chunks=[chunk],
        collection="docs",
        doc_id="doc-bridge",
        source_path="bridge.txt",
    )

    class VectorStore:
        def get_by_ids(self, ids, trace=None, **kwargs):
            return [
                {
                    "id": "c1",
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                for _ in ids
            ]

    retriever = SparseRetriever(
        bm25_indexer=indexer,
        vector_store=VectorStore(),
        default_collection="docs",
    )

    results = retriever.retrieve(["fts5"], top_k=3)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert "sparse backend" in results[0].text
