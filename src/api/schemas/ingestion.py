"""Ingestion job API schemas."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class IngestionJobCreateRequest(BaseModel):
    file_path: Optional[str] = Field(default=None, max_length=4000)
    content: Optional[str] = None
    filename: Optional[str] = Field(default=None, max_length=500)
    collection_id: str = Field(default="default", max_length=200)
    force: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
