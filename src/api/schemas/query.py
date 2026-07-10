"""Query API schemas."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query")
    collection_id: Optional[str] = Field(default=None, max_length=200)
    top_k: int = Field(default=5, ge=1, le=20)
    use_rerank: bool = True
    use_llm: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()
