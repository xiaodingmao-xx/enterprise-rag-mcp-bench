"""Offline tests for document versioning, cleanup and audit behavior."""

from __future__ import annotations

from src.ingestion.document_manager import DocumentManager
from src.ingestion.storage.audit_log_store import SQLiteAuditLogStore
from src.ingestion.storage.index_cleaner import NoopIndexCleaner
from src.ingestion.storage.sqlite_document_version_store import SQLiteDocumentVersionStore


def _store(tmp_path):
    audit = SQLiteAuditLogStore(tmp_path / "audit.db")
    return SQLiteDocumentVersionStore(tmp_path / "versions.db", audit_log_store=audit), audit


def _version(store, content_hash: str):
    record = store.get_or_create_record(
        tenant_id="tenant-a",
        source_id="wiki",
        external_document_id="doc-1",
        title="Policy",
    )
    version = store.create_version(
        document_id=record.document_id,
        content_hash=content_hash,
        metadata_hash="metadata-" + content_hash,
        parser_version="parser-1",
        chunker_version="chunker-1",
        embedding_model="embed-1",
    )
    return record, version


def test_same_fingerprint_is_idempotent(tmp_path):
    store, _ = _store(tmp_path)
    record, first = _version(store, "hash-1")
    second = store.create_version(
        document_id=record.document_id,
        content_hash="hash-1",
        metadata_hash="metadata-hash-1",
        parser_version="parser-1",
        chunker_version="chunker-1",
        embedding_model="embed-1",
    )
    assert first.version_id == second.version_id
    assert len(store.list_versions(record.document_id)) == 1


def test_content_change_creates_new_version_and_preserves_old(tmp_path):
    store, _ = _store(tmp_path)
    record, first = _version(store, "hash-1")
    store.mark_version_active(first.version_id)
    store.activate_version(record.document_id, first.version_id)
    second = store.create_version(
        document_id=record.document_id,
        content_hash="hash-2",
        metadata_hash="metadata-hash-2",
        parser_version="parser-1",
        chunker_version="chunker-1",
        embedding_model="embed-1",
    )
    store.mark_version_active(second.version_id)
    current = store.activate_version(record.document_id, second.version_id)
    assert current.current_version_id == second.version_id
    assert len(store.list_versions(record.document_id)) == 2
    assert store.list_versions(record.document_id)[0].status == "active"


def test_rollback_switches_pointer_without_deleting_versions(tmp_path):
    store, audit = _store(tmp_path)
    record, first = _version(store, "hash-1")
    store.mark_version_active(first.version_id)
    store.activate_version(record.document_id, first.version_id, tenant_id="tenant-a", actor="alice")
    second = store.create_version(
        document_id=record.document_id,
        content_hash="hash-2",
        metadata_hash="metadata-hash-2",
        parser_version="parser-1",
        chunker_version="chunker-1",
        embedding_model="embed-1",
    )
    store.mark_version_active(second.version_id)
    store.activate_version(record.document_id, second.version_id)
    rolled_back = store.rollback_to_version(record.document_id, first.version_id, tenant_id="tenant-a", actor="alice")
    assert rolled_back.current_version_id == first.version_id
    assert {item.status for item in store.list_versions(record.document_id)} == {"active"}
    assert any(item["action"] == "activate_version" for item in audit.list(document_id=record.document_id))


def test_delete_current_version_cleans_indexes_and_audits(tmp_path):
    store, audit = _store(tmp_path)

    class Cleaner(NoopIndexCleaner):
        def __init__(self):
            self.calls = []

        def delete_vectors(self, document_id, version_id): self.calls.append(("vectors", document_id, version_id))
        def delete_sparse_index(self, document_id, version_id): self.calls.append(("sparse", document_id, version_id))
        def delete_images(self, document_id, version_id): self.calls.append(("images", document_id, version_id))
        def delete_cache(self, document_id, version_id): self.calls.append(("cache", document_id, version_id))

    record, version = _version(store, "hash-1")
    store.mark_version_active(version.version_id)
    store.activate_version(record.document_id, version.version_id)
    cleaner = Cleaner()
    manager = DocumentManager(None, None, None, None, store, cleaner, audit)
    deleted = manager.delete_current_version(record.document_id, tenant_id="tenant-a", actor="alice")
    assert deleted.status == "deleted"
    assert len(cleaner.calls) == 4
    assert audit.list(document_id=record.document_id)[0]["action"] == "delete_version"

