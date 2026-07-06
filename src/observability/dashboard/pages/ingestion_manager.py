"""Ingestion Manager page – upload files, trigger ingestion, delete documents.

Layout:
1. File uploader + collection selector
2. Ingest button → progress bar (using on_progress callback)
3. Document list with delete buttons
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


@st.cache_resource
def _get_ingestion_queue():
    from src.core.settings import load_settings
    from src.ingestion.task_queue import IngestionTaskQueue

    return IngestionTaskQueue(settings=load_settings())


def _safe_upload_name(name: str) -> str:
    path = Path(name)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    suffix = path.suffix.lower()
    return f"{stem or 'upload'}{suffix}"


def _save_uploaded_file(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
) -> Path:
    queue = _get_ingestion_queue()
    upload_dir = queue.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_upload_name(uploaded_file.name)
    target = upload_dir / f"{uuid.uuid4().hex}_{safe_name}"
    with target.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return target


def _run_ingestion(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
) -> None:
    """Save the uploaded file to a temp location and run the pipeline."""
    from src.core.settings import load_settings
    from src.core.trace import TraceContext, TraceCollector
    from src.ingestion.pipeline import IngestionPipeline
    from src.libs.loader.document_quality import DOCUMENT_QUALITY_REJECTION_MESSAGE

    settings = load_settings()

    # Write uploaded file to a temp location
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    _STAGE_LABELS = {
        "integrity": "🔍 Checking file integrity…",
        "quality": "Checking document quality...",
        "load": "📄 Loading document…",
        "split": "✂️ Chunking document…",
        "transform": "🔄 Transforming chunks (LLM refine + enrich)…",
        "embed": "🔢 Encoding vectors…",
        "upsert": "💾 Storing to database…",
    }

    def on_progress(stage: str, current: int, total: int) -> None:
        frac = (current - 1) / total  # stage just started, show partial progress
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(frac, text=f"[{current}/{total}] {label}")
        status_text.caption(label)

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = collection
    trace.metadata["source"] = "dashboard"

    try:
        pipeline = IngestionPipeline(settings, collection=collection)
        result = pipeline.run(
            file_path=tmp_path,
            trace=trace,
            on_progress=on_progress,
        )
        if not result.success:
            progress_bar.progress(1.0, text="Failed")
            if result.error == DOCUMENT_QUALITY_REJECTION_MESSAGE:
                status_text.error(DOCUMENT_QUALITY_REJECTION_MESSAGE)
            else:
                status_text.error(f"Ingestion failed: {result.error}")
            return

        skipped = result.stages.get("integrity", {}).get("skipped", False)
        if skipped:
            progress_bar.progress(1.0, text="Already processed")
            status_text.info(
                f"**{uploaded_file.name}** 文件已经处理过，已跳过重复摄取。"
            )
            return

        progress_bar.progress(1.0, text="✅ Complete")
        status_text.success(f"Successfully ingested **{uploaded_file.name}** into collection **{collection}**.")
    except Exception as exc:
        status_text.error(f"Ingestion failed: {exc}")
    finally:
        TraceCollector().collect(trace)
        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def render() -> None:
    """Render the Ingestion Manager page."""
    st.header("📥 Ingestion Manager")

    # ── Upload section ─────────────────────────────────────────────
    st.subheader("📤 Upload & Ingest")

    queue = _get_ingestion_queue()

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Select files to ingest",
            type=["pdf", "txt", "md", "docx"],
            key="ingest_uploader",
            accept_multiple_files=True,
        )
    with col2:
        collection = st.text_input("Collection", value="default", key="ingest_collection")
        force = st.checkbox("Force reprocess", value=False, key="ingest_force")

    if uploaded_files:
        if st.button("Submit Ingestion Jobs", key="btn_ingest_async"):
            target_collection = collection.strip() or "default"
            for uploaded in uploaded_files:
                stored_path = _save_uploaded_file(uploaded)
                job_id = queue.submit_file(
                    stored_path,
                    collection=target_collection,
                    force=force,
                    original_name=uploaded.name,
                )
                st.success(f"Submitted **{uploaded.name}** as job `{job_id}`.")

    st.subheader("Ingestion Jobs")
    if st.button("Refresh Jobs", key="btn_refresh_jobs"):
        st.rerun()
    jobs = queue.list_jobs(limit=20)
    if jobs:
        rows = [
            {
                "job_id": job["job_id"][:12],
                "status": job["status"],
                "file": job.get("original_name") or Path(job["file_path"]).name,
                "collection": job["collection"],
                "stage": job.get("progress_stage") or "",
                "progress": (
                    f"{job.get('progress_current', 0)}/{job.get('progress_total', 0)}"
                    if job.get("progress_total")
                    else ""
                ),
                "error": (job.get("error") or "")[:120],
            }
            for job in jobs
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No ingestion jobs yet.")

    st.divider()

    # ── Document management section ────────────────────────────────
    st.subheader("🗑️ Manage Documents")

    try:
        svc = DataService()
        docs = svc.list_documents()
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if not docs:
        st.info(
            "**No documents ingested yet.** "
            "Upload a PDF, TXT, MD, or DOCX file above and click \"Submit Ingestion Jobs\" to begin."
        )
        return

    for idx, doc in enumerate(docs):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"**{doc['source_path']}** — "
                f"collection: `{doc.get('collection', '—')}` | "
                f"chunks: {doc['chunk_count']} | "
                f"images: {doc['image_count']}"
            )
        with col_btn:
            if st.button("🗑️ Delete", key=f"del_{idx}"):
                try:
                    result = svc.delete_document(
                        source_path=doc["source_path"],
                        collection=doc.get("collection", "default"),
                        source_hash=doc.get("source_hash"),
                    )
                    if result.success:
                        st.success(
                            f"Deleted: {result.chunks_deleted} chunks, "
                            f"{result.images_deleted} images removed."
                        )
                        st.rerun()
                    else:
                        st.warning(f"Partial delete. Errors: {result.errors}")
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
