"""Core ingestion-job facade backed by the existing local task queue."""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any

from src.api.errors import APIError
from src.core.settings import Settings, resolve_path
from src.security.context import RequestContext


class IngestionService:
    def __init__(self, settings: Settings | Any = None, *, queue: Any = None) -> None:
        self.settings = settings
        self.queue = queue
        self._jobs: dict[str, dict[str, Any]] = {}

    def _get_queue(self) -> Any:
        if self.queue is None:
            from src.ingestion.task_queue import get_default_ingestion_queue

            self.queue = get_default_ingestion_queue(self.settings)
        return self.queue

    async def create_job(self, payload: dict[str, Any], context: RequestContext) -> dict[str, Any]:
        content = payload.get("content")
        file_path = payload.get("file_path")
        if not content and not file_path:
            raise APIError("VALIDATION_ERROR", "Either content or file_path is required")
        collection = str(payload.get("collection_id") or "default")
        job_id = uuid.uuid4().hex
        record = {
            "job_id": job_id,
            "tenant_id": str(context.tenant_id or "local"),
            "collection": collection,
            "status": "queued",
            "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }
        self._jobs[job_id] = record

        try:
            queue = self._get_queue()
            if content:
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(payload.get("filename") or "document.txt"))
                upload_dir = Path(getattr(queue, "upload_dir", resolve_path("./data/uploads/ingestion")))
                upload_dir.mkdir(parents=True, exist_ok=True)
                path = upload_dir / f"{job_id}_{safe_name}"
                path.write_text(str(content), encoding="utf-8")
            else:
                path = Path(str(file_path))
                if not path.is_file():
                    raise APIError("INGESTION_FAILED", "The ingestion source file was not found")
            queue_job_id = await asyncio.to_thread(
                queue.submit_file,
                path,
                collection=collection,
                force=bool(payload.get("force", False)),
                original_name=str(payload.get("filename") or path.name),
            )
            if queue_job_id != job_id:
                self._jobs[queue_job_id] = {**record, "job_id": queue_job_id}
                self._jobs.pop(job_id, None)
                job_id = queue_job_id
        except APIError:
            self._jobs[job_id]["status"] = "failed"
            raise
        except Exception as exc:
            self._jobs[job_id]["status"] = "failed"
            raise APIError("INGESTION_FAILED", "Unable to create ingestion job") from exc
        return dict(self._jobs[job_id])

    async def get_job(self, job_id: str, context: RequestContext) -> dict[str, Any]:
        record = self._jobs.get(job_id)
        if record is None and self.queue is not None:
            raw = await asyncio.to_thread(self.queue.get_job, job_id)
            if raw is not None:
                record = {**raw, "tenant_id": str(context.tenant_id or "local")}
        if record is None:
            raise APIError("INGESTION_JOB_NOT_FOUND", "Ingestion job not found")
        if str(record.get("tenant_id")) != str(context.tenant_id):
            raise APIError("TENANT_MISMATCH", "The ingestion job belongs to another tenant")
        if self.queue is not None:
            raw = await asyncio.to_thread(self.queue.get_job, job_id)
            if raw:
                record = {**record, **raw}
        return dict(record)

    async def list_jobs(self, context: RequestContext, *, limit: int = 50) -> list[dict[str, Any]]:
        return [
            item
            for item in self._jobs.values()
            if str(item.get("tenant_id")) == str(context.tenant_id)
        ][:limit]
