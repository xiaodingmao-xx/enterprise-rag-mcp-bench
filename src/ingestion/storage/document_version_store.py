"""Contracts and records for logical documents and immutable versions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    tenant_id: str
    source_id: str
    external_document_id: str
    current_version_id: Optional[str] = None
    title: str = ""
    source_uri: str = ""
    source_type: str = ""
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    deleted_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentVersion:
    version_id: str
    document_id: str
    content_hash: str
    metadata_hash: str
    parser_version: str
    chunker_version: str
    embedding_model: str
    created_at: str = ""
    status: str = "processing"
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DocumentVersionStore(ABC):
    """Backend-neutral document/version lifecycle contract."""

    @abstractmethod
    def get_or_create_record(self, **kwargs: Any) -> DocumentRecord:
        raise NotImplementedError

    @abstractmethod
    def find_existing_version(self, **kwargs: Any) -> Optional[DocumentVersion]:
        raise NotImplementedError

    @abstractmethod
    def create_version(self, **kwargs: Any) -> DocumentVersion:
        raise NotImplementedError

    @abstractmethod
    def mark_version_active(self, version_id: str) -> DocumentVersion:
        raise NotImplementedError

    @abstractmethod
    def mark_version_failed(self, version_id: str, error_message: str) -> DocumentVersion:
        raise NotImplementedError

    @abstractmethod
    def activate_version(self, document_id: str, version_id: str, **kwargs: Any) -> DocumentRecord:
        raise NotImplementedError

    @abstractmethod
    def rollback_to_version(self, document_id: str, version_id: str, **kwargs: Any) -> DocumentRecord:
        raise NotImplementedError

