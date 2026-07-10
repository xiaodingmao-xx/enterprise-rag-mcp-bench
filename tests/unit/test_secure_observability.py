"""Tests for secure traces, audit records, and log retention."""

import json
import os
import time

import pytest

from src.core.trace.trace_context import TraceContext
from src.observability.audit import AuditLogger
from src.observability.redaction import RedactionConfig, redact_text
from src.observability.retention import LogRetentionManager


def test_api_key_email_and_phone_are_redacted() -> None:
    value = "api_key=sk-abcdefghijklmnopqrstuvwxyz email=a@example.com phone=13812345678"
    result = redact_text(value)

    assert "sk-abcdefghijklmnopqrstuvwxyz" not in result
    assert "a@example.com" not in result
    assert "13812345678" not in result


def test_default_trace_contains_no_complete_chunk_or_prompt() -> None:
    chunk = "FULL-CHUNK-CONTENT " * 100
    prompt = "FULL-PROMPT-CONTENT " * 100
    trace = TraceContext()
    trace.record_stage("retrieval", {"chunks": [{"chunk_id": "c1", "text": chunk}], "prompt": prompt})

    payload = json.dumps(trace.to_dict(), ensure_ascii=False)

    assert chunk not in payload
    assert prompt not in payload
    assert "redacted_preview" in payload
    assert "prompt_redacted" in payload


def test_audit_log_contains_tenant_user_and_operation(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    event = AuditLogger(path=path).write(
        trace_id="trace-1",
        request_id="request-1",
        tenant_id="tenant-a",
        user_id="alice",
        operation="query_knowledge_hub",
        resource="contracts",
        tool="query_knowledge_hub",
        document_ids=["doc-1"],
    )

    stored = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert event["tenant_id"] == "tenant-a"
    assert stored["tenant_id"] == "tenant-a"
    assert stored["user_id_hash"].startswith("sha256:")
    assert stored["operation"] == "query_knowledge_hub"


def test_retention_deletes_expired_trace_file(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    old = time.time() - 3 * 86400
    os.utime(path, (old, old))

    deleted = LogRetentionManager(max_days=1).cleanup_expired([path])

    assert deleted == 1
    assert not path.exists()


def test_redaction_disabled_is_development_only() -> None:
    RedactionConfig(enabled=False, environment="development")
    with pytest.raises(ValueError):
        RedactionConfig(enabled=False, environment="production")
    with pytest.raises(ValueError):
        RedactionConfig(include_prompt=True, environment="production")

