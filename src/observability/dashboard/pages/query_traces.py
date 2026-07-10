"""Safe query trace view.

Only persistence-time redacted fields are rendered.  This page intentionally
does not offer a raw-query or raw-chunk viewer.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService
from src.observability.redaction import redact_text


def render() -> None:
    st.header("Query Traces")
    service = TraceService()
    traces = service.list_traces(trace_type="query")
    if not traces:
        st.info("No query traces recorded yet. Run a query first!")
        return

    st.subheader(f"Query history ({len(traces)})")
    rows = service.trace_rows(limit=len(traces))
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )

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

    query_preview = metadata.get("query_preview", metadata.get("redacted_preview", ""))
    if query_preview:
        st.caption("Query preview (redacted/truncated)")
        st.info(redact_text(query_preview, max_length=256))

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


def _render_chunk_list(chunks: List[Dict[str, Any]], prefix: str = "chunk") -> None:
    """Backward-compatible helper that renders only safe previews."""

    for index, chunk in enumerate(chunks):
        st.text_area(
            f"{prefix}_{index}",
            value=redact_text(chunk.get("redacted_preview", chunk.get("text", "")), max_length=256),
            height=100,
            disabled=True,
            label_visibility="collapsed",
        )

