"""Data Browser page – browse ingested documents, chunks, and images.

Layout:
1. Collection selector (sidebar)
2. Document list with chunk counts
3. Expandable document detail → chunk cards with text + metadata
4. Image preview gallery
"""

from __future__ import annotations

import math
from pathlib import Path

import streamlit as st
from PIL import Image, ImageStat, UnidentifiedImageError

from src.observability.dashboard.services.data_service import DataService


IMAGE_PREVIEW_WIDTH = 200
MAX_PREVIEW_ASPECT_RATIO = 8.0
BLACK_PIXEL_THRESHOLD = 8
BLACK_PIXEL_RATIO = 0.985
BLACK_MEAN_THRESHOLD = 4.0


def _is_nearly_black_image(image: Image.Image) -> bool:
    """Return True for solid or near-solid black images."""
    sample = image.convert("L")
    sample.thumbnail((64, 64))

    histogram = sample.histogram()
    total_pixels = max(1, sample.width * sample.height)
    dark_pixels = sum(histogram[: BLACK_PIXEL_THRESHOLD + 1])
    dark_ratio = dark_pixels / total_pixels
    mean_luminance = ImageStat.Stat(sample).mean[0]

    return (
        dark_ratio >= BLACK_PIXEL_RATIO
        and mean_luminance <= BLACK_MEAN_THRESHOLD
    )


def _image_preview_error(
    image_path: Path,
    *,
    preview_width: int = IMAGE_PREVIEW_WIDTH,
) -> str | None:
    """Return a user-facing error if an image cannot be safely previewed."""
    if not image_path.exists():
        return "file missing"
    if preview_width <= 0:
        return "invalid preview width"

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                return f"invalid image size: {width}x{height}"

            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > MAX_PREVIEW_ASPECT_RATIO:
                return f"image aspect ratio too long: {width}x{height}"

            preview_height = math.floor(height * preview_width / width)
            if preview_height <= 0:
                return (
                    f"image aspect ratio too wide for {preview_width}px preview: "
                    f"{width}x{height}"
                )

            image.load()
            if _is_nearly_black_image(image):
                return f"near-black image skipped: {width}x{height}"
    except (
        OSError,
        UnidentifiedImageError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        return f"invalid image: {exc}"

    return None


def _render_image_preview(img: dict, *, width: int = IMAGE_PREVIEW_WIDTH) -> None:
    """Render one image preview without letting bad files crash the page."""
    image_id = str(img.get("image_id", "image"))
    img_path = Path(str(img.get("file_path", "")))
    error = _image_preview_error(img_path, preview_width=width)
    if error is not None:
        st.caption(f"{image_id} ({error})")
        return

    try:
        st.image(str(img_path), caption=image_id, width=width)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        st.caption(f"{image_id} (preview failed: {exc})")


def render() -> None:
    """Render the Data Browser page."""
    st.header("🔍 Data Browser")

    try:
        svc = DataService()
    except Exception as exc:
        st.error(f"Failed to initialise DataService: {exc}")
        return

    # ── Collection selector ────────────────────────────────────────
    collections = svc.list_collections()
    if "default" not in collections:
        collections.insert(0, "default")
    collection = st.selectbox(
        "Collection",
        options=collections,
        index=0,
        key="db_collection_filter",
    )
    coll_arg = collection if collection else None

    # ── Danger zone: clear all data ────────────────────────────────
    st.divider()
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.warning(
            "This will **permanently delete** all data: "
            "ChromaDB collections, BM25 indexes, images, ingestion history, and trace logs."
        )
        col_btn, col_status = st.columns([1, 2])
        with col_btn:
            if st.button("🗑️ Clear All Data", type="primary", key="btn_clear_all"):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.error("Are you sure? This action cannot be undone!")
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button("✅ Yes, delete everything", key="btn_confirm_clear"):
                    result = svc.reset_all()
                    st.session_state["confirm_clear"] = False
                    if result["errors"]:
                        st.warning(
                            f"Cleared with {len(result['errors'])} error(s): "
                            + "; ".join(result["errors"])
                        )
                    else:
                        st.success(
                            f"All data cleared! "
                            f"{result['collections_deleted']} collection(s) deleted."
                        )
                    st.rerun()
            with c2:
                if st.button("❌ Cancel", key="btn_cancel_clear"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

    st.divider()

    # ── Document list ──────────────────────────────────────────────
    try:
        docs = svc.list_documents(coll_arg)
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if not docs:
        st.info(
            "**No documents found in this collection.** "
            "Use the Ingestion Manager page to upload and ingest files, "
            "or select a different collection from the dropdown above."
        )
        return

    st.subheader(f"📄 Documents ({len(docs)})")

    for idx, doc in enumerate(docs):
        source_name = Path(doc["source_path"]).name
        label = f"📑 {source_name}  —  {doc['chunk_count']} chunks · {doc['image_count']} images"
        with st.expander(label, expanded=(len(docs) == 1)):
            # ── Document metadata ──────────────────────────────────
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Chunks", doc["chunk_count"])
            col_b.metric("Images", doc["image_count"])
            col_c.metric("Collection", doc.get("collection", "—"))
            st.caption(
                f"**Source:** {doc['source_path']}  ·  "
                f"**Hash:** `{doc['source_hash'][:16]}…`  ·  "
                f"**Processed:** {doc.get('processed_at', '—')}"
            )

            st.divider()

            # ── Chunk cards ────────────────────────────────────────
            chunks = svc.get_chunks(doc["source_hash"], coll_arg)
            if chunks:
                st.markdown(f"### 📦 Chunks ({len(chunks)})")
                for cidx, chunk in enumerate(chunks):
                    text = chunk.get("text", "")
                    meta = chunk.get("metadata", {})
                    chunk_id = chunk["id"]

                    # Title from metadata or first line
                    title = meta.get("title", "")
                    if not title:
                        title = text[:60].replace("\n", " ").strip()
                        if len(text) > 60:
                            title += "…"

                    with st.container(border=True):
                        st.markdown(
                            f"**Chunk {cidx + 1}** · `{chunk_id[-16:]}` · "
                            f"{len(text)} chars"
                        )
                        # Show the actual chunk text (scrollable)
                        _height = max(120, min(len(text) // 2, 600))
                        st.text_area(
                            "Content",
                            value=text,
                            height=_height,
                            disabled=True,
                            key=f"chunk_text_{idx}_{cidx}",
                            label_visibility="collapsed",
                        )
                        # Expandable metadata
                        with st.expander("📋 Metadata", expanded=False):
                            st.json(meta)
            else:
                st.caption("No chunks found in vector store for this document.")

            # ── Image preview ──────────────────────────────────────
            images = svc.get_images(doc["source_hash"], coll_arg)
            if images:
                st.divider()
                st.markdown(f"### 🖼️ Images ({len(images)})")
                img_cols = st.columns(min(len(images), 4))
                for iidx, img in enumerate(images):
                    with img_cols[iidx % len(img_cols)]:
                        _render_image_preview(img)
