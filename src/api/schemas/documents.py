"""Document API schemas."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class DocumentCreateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=500)
    source_uri: Optional[str] = Field(default=None, max_length=2000)
    source_type: str = Field(default="text", max_length=100)
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collection_id: Optional[str] = Field(default=None, max_length=200)
    external_document_id: Optional[str] = Field(default=None, max_length=500)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("content must not be blank when provided")
        return value


class DocumentRollbackRequest(BaseModel):
    version_id: str = Field(..., min_length=1, max_length=200)
