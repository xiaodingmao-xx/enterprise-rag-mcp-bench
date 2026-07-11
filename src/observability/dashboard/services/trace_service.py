"""TraceService – read and parse traces from logs/traces.jsonl.

Provides a typed, filterable interface over the raw JSONL trace log.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.core.settings import resolve_path
from src.observability.retention import LogRetentionManager

logger = logging.getLogger(__name__)

# Default path to the traces file (absolute, CWD-independent)
DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")
DEFAULT_AUDIT_PATH = resolve_path("logs/audit.jsonl")
CHINA_TIMEZONE = ZoneInfo("Asia/Shanghai")


class TraceService:
    """Read-only service for querying recorded traces.

    Args:
        traces_path: Path to the JSONL file.  Defaults to
            ``logs/traces.jsonl``.
    """

    def __init__(
        self,
        traces_path: Optional[str | Path] = None,
        audit_path: Optional[str | Path] = None,
        image_index_path: str | Path | None = None,
        images_root: str | Path | None = None,
    ) -> None:
        self.traces_path = Path(traces_path) if traces_path else DEFAULT_TRACES_PATH
        self.audit_path = Path(audit_path) if audit_path else DEFAULT_AUDIT_PATH
        self.image_index_path = (
            Path(image_index_path)
            if image_index_path
            else resolve_path("data/db/image_index.db")
        )
        self.images_root = (
            Path(images_root) if images_root else resolve_path("data/images")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_traces(
        self,
        trace_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return traces in reverse-chronological order.

        Args:
            trace_type: Filter by ``trace_type`` field (e.g.
                ``"ingestion"`` or ``"query"``).  ``None`` = all.
            limit: Maximum number of traces to return.

        Returns:
            List of trace dicts (newest first).
        """
        traces = self._load_all()

        if trace_type:
            traces = [t for t in traces if t.get("trace_type") == trace_type]

        # Newest first
        traces.sort(key=lambda t: t.get("started_at", ""), reverse=True)

        return traces[:limit]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single trace by its ``trace_id``.

        Returns:
            Trace dict, or ``None`` if not found.
        """
        for t in self._load_all():
            if t.get("trace_id") == trace_id:
                return t
        return None

    def delete_trace(self, trace_id: str) -> bool:
        """Delete one debug trace record by id; audit records remain untouched."""

        return LogRetentionManager().delete_trace(self.traces_path, trace_id)

    def get_trace_images(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Return safely validated images associated with an ingestion trace.

        Image paths in a trace are diagnostic metadata and must not be trusted
        directly. This method uses the SQLite image index as the source of
        truth, then accepts only existing files located under ``images_root``.
        Older traces that lack the ``upsert.image_store.images`` payload fall
        back to the document hash from their integrity stage.
        """
        trace_images = self._trace_image_entries(trace)
        document_hash = self._trace_document_hash(trace, trace_images)
        indexed = self._load_indexed_images(
            image_ids=[item["image_id"] for item in trace_images],
            document_hash=document_hash,
        )

        # Prefer trace order: it reflects extraction order and preserves page
        # metadata from the original ingestion run.
        if trace_images:
            results: list[dict[str, Any]] = []
            for item in trace_images:
                indexed_item = indexed.get(item["image_id"])
                if indexed_item is None:
                    results.append({
                        "image_id": item["image_id"],
                        "page_num": item.get("page_num"),
                        "file_path": "",
                        "error": "not registered in image index",
                    })
                else:
                    results.append(self._validate_indexed_image(indexed_item))
            return results

        # Old traces have no embedded image list. The image index is the best
        # available source for all images belonging to the document.
        return [self._validate_indexed_image(item) for item in indexed.values()]

    def list_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read audit events from the separated audit stream."""

        records = self._load_jsonl(self.audit_path)
        records.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return records[:limit]

    def trace_rows(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Flatten safe trace data into Dashboard rows without raw document text."""

        rows: List[Dict[str, Any]] = []
        for trace in self.list_traces(limit=limit):
            base = {
                "trace_id": trace.get("trace_id", ""),
                "status": trace.get("status", ""),
                "error_code": trace.get("error_code"),
                "duration": trace.get("latency_ms", trace.get("total_elapsed_ms", 0)),
            }
            metadata = trace.get("metadata", {}) if isinstance(trace.get("metadata"), dict) else {}
            default_document_id = metadata.get("document_id") or metadata.get("doc_id")
            emitted = False
            for stage in trace.get("stages", []) or []:
                stage_name = stage.get("stage", "")
                stage_data = stage.get("data", {}) if isinstance(stage.get("data"), dict) else {}
                chunks = stage_data.get("chunks") if isinstance(stage_data.get("chunks"), list) else []
                stage_document_id = stage_data.get("document_id") or stage_data.get("doc_id") or default_document_id
                if chunks:
                    for chunk in chunks:
                        if not isinstance(chunk, dict):
                            continue
                        rows.append({
                            **base,
                            "stage": stage_name,
                            "duration": stage.get("elapsed_ms", base["duration"]),
                            "document_id": chunk.get("document_id") or stage_document_id or "",
                            "chunk_id": chunk.get("chunk_id", ""),
                            "redacted_preview": (
                                chunk.get("redacted_preview")
                                or chunk.get("text_before_preview")
                                or chunk.get("text_after_preview")
                                or ""
                            ),
                        })
                        emitted = True
                if not chunks:
                    rows.append({
                        **base,
                        "stage": stage_name,
                        "duration": stage.get("elapsed_ms", base["duration"]),
                        "document_id": stage_document_id or "",
                        "chunk_id": stage_data.get("chunk_id", ""),
                        "redacted_preview": stage_data.get("redacted_preview", ""),
                    })
                    emitted = True
            if not emitted:
                rows.append({**base, "stage": trace.get("operation", ""), "document_id": default_document_id or "", "chunk_id": "", "redacted_preview": ""})
        return rows[:limit]

    def get_stage_timings(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract stage timings from a trace.

        Returns:
            List of dicts with keys: stage_name, elapsed_ms, data.
            Ordered by appearance.
        """
        stages = trace.get("stages", [])
        timings: List[Dict[str, Any]] = []
        for s in stages:
            # The raw stage dict has: stage, timestamp, data (dict), elapsed_ms
            # Extract the inner 'data' dict directly rather than flattening
            stage_data = s.get("data", {})
            if not isinstance(stage_data, dict):
                stage_data = {}
            if not stage_data:
                stage_data = {
                    key: value
                    for key, value in s.items()
                    if key not in {"stage", "timestamp", "elapsed_ms"}
                }
            timings.append(
                {
                    "stage_name": s.get("stage"),
                    "elapsed_ms": s.get("elapsed_ms", 0),
                    "data": stage_data,
                }
            )
        return timings

    @staticmethod
    def format_china_time(value: Any) -> str:
        """Format an ISO timestamp as China local time.

        Trace files store timestamps in UTC.  Dashboard pages should present
        them in China time so recent events match the user's local clock.
        """
        if not value:
            return "—"

        raw = str(value)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return raw[:19]

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(CHINA_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trace_image_entries(trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract image identifiers recorded by the ingestion upsert stage."""
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for stage in trace.get("stages", []) or []:
            if not isinstance(stage, dict) or stage.get("stage") != "upsert":
                continue
            data = stage.get("data", {})
            if not isinstance(data, dict):
                continue
            image_store = data.get("image_store", {})
            images = image_store.get("images", []) if isinstance(image_store, dict) else []
            for image in images if isinstance(images, list) else []:
                if not isinstance(image, dict):
                    continue
                image_id = str(image.get("image_id", "")).strip()
                if not image_id or image_id in seen:
                    continue
                seen.add(image_id)
                entries.append({
                    "image_id": image_id,
                    "page_num": image.get("page"),
                    "doc_hash": image.get("doc_hash"),
                })
        return entries

    @staticmethod
    def _trace_document_hash(
        trace: dict[str, Any], trace_images: list[dict[str, Any]]
    ) -> str | None:
        """Find the document hash needed to support older trace formats."""
        for image in trace_images:
            doc_hash = image.get("doc_hash")
            if doc_hash:
                return str(doc_hash)
        for stage in trace.get("stages", []) or []:
            if not isinstance(stage, dict) or stage.get("stage") != "integrity":
                continue
            data = stage.get("data", {})
            if isinstance(data, dict) and data.get("file_hash"):
                return str(data["file_hash"])
        return None

    def _load_indexed_images(
        self, *, image_ids: list[str], document_hash: str | None
    ) -> dict[str, dict[str, Any]]:
        """Load registered images without creating or mutating the index."""
        if not self.image_index_path.exists():
            return {}

        clauses: list[str] = []
        params: list[str] = []
        if image_ids:
            placeholders = ", ".join("?" for _ in image_ids)
            clauses.append(f"image_id IN ({placeholders})")
            params.extend(image_ids)
        if document_hash:
            clauses.append("doc_hash = ?")
            params.append(document_hash)
        if not clauses:
            return {}

        query = (
            "SELECT image_id, file_path, collection, doc_hash, page_num "
            "FROM image_index WHERE " + " OR ".join(clauses) + " ORDER BY created_at ASC"
        )
        try:
            with sqlite3.connect(self.image_index_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            logger.warning("Failed to load trace images from %s: %s", self.image_index_path, exc)
            return {}
        return {str(row["image_id"]): dict(row) for row in rows}

    def _validate_indexed_image(self, image: dict[str, Any]) -> dict[str, Any]:
        """Return a render-safe image record, retaining a diagnostic error."""
        result = {
            "image_id": str(image.get("image_id", "")),
            "page_num": image.get("page_num"),
            "file_path": str(image.get("file_path", "")),
            "error": None,
        }
        try:
            images_root = self.images_root.resolve()
            image_path = Path(result["file_path"]).resolve()
            image_path.relative_to(images_root)
        except (OSError, ValueError):
            result["error"] = "image path is outside the configured image storage"
            return result

        if not image_path.is_file():
            result["error"] = "image file is missing"
        return result

    def _load_all(self) -> List[Dict[str, Any]]:
        """Parse every line in the JSONL file.

        Silently skips malformed lines.
        """
        if not self.traces_path.exists():
            return []

        return self._load_jsonl(self.traces_path)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed trace line: %s", line[:80])
        return records
