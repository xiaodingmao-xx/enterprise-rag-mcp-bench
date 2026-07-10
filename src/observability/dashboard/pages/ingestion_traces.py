"""Safe ingestion trace view for enterprise documents."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService
from src.observability.redaction import redact_text


def _is_already_processed_trace(trace: Dict[str, Any], stages_by_name: Dict[str, Any]) -> bool:
    """Return whether an ingestion trace was intentionally skipped as a duplicate."""

    metadata = trace.get("metadata", {}) if isinstance(trace, dict) else {}
    if isinstance(metadata, dict) and metadata.get("skip_reason") == "already_processed":
        return True
    integrity = stages_by_name.get("integrity", {}) if isinstance(stages_by_name, dict) else {}
    data = integrity.get("data", {}) if isinstance(integrity, dict) else {}
    return (
        isinstance(data, dict)
        and data.get("skipped") is True
        and data.get("reason") == "already_processed"
    )


def render() -> None:
    st.header("Ingestion Traces")
    service = TraceService()
    traces = service.list_traces(trace_type="ingestion")
    if not traces:
        st.info("No ingestion traces recorded yet. Run an ingestion first!")
        return

    st.subheader(f"Trace history ({len(traces)})")
    st.dataframe(service.trace_rows(limit=len(traces)), use_container_width=True, hide_index=True)

    for index, trace in enumerate(traces):
        trace_id = str(trace.get("trace_id", "unknown"))
        duration = trace.get("latency_ms", trace.get("total_elapsed_ms", 0))
        status = trace.get("status", "unknown")
        with st.expander(f"{trace_id} | {duration} ms | {status}", expanded=index == 0):
            _render_trace_summary(trace, service)


def _render_trace_summary(trace: Dict[str, Any], service: TraceService) -> None:
    metadata = trace.get("metadata", {}) if isinstance(trace.get("metadata"), dict) else {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Trace ID", trace.get("trace_id", ""))
    with c2:
        st.metric("Tenant", trace.get("tenant_id", ""))
    with c3:
        st.metric("Status", trace.get("status", "unknown"))
    with c4:
        st.metric("Error code", trace.get("error_code") or "—")

    source = metadata.get("source_path", metadata.get("source_path_preview", ""))
    if source:
        st.caption(f"Source path (user name redacted): {redact_text(source, max_length=256, is_path=True)}")

    trace_id = trace.get("trace_id", "")
    rows = [row for row in service.trace_rows(limit=1000) if row.get("trace_id") == trace_id]
    st.dataframe(
        [
            {
                "trace_id": row.get("trace_id", ""),
                "stage": row.get("stage", ""),
                "duration": row.get("duration", 0),
                "status": row.get("status", ""),
                "error_code": row.get("error_code"),
                "document_id": row.get("document_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "redacted_preview": redact_text(row.get("redacted_preview", ""), max_length=256),
            }
            for row in rows
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_load_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """Compatibility renderer: document body is always bounded and redacted."""

    st.metric("Doc ID", str(data.get("doc_id", ""))[:16])
    st.metric("Text Length", f"{data.get('text_length', 0):,}")
    preview = data.get("redacted_preview", data.get("text_preview", ""))
    if preview:
        st.caption("Redacted preview")
        st.text_area(f"load_preview_{trace_idx}", redact_text(preview, max_length=256), height=100, disabled=True)
