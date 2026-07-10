"""
Ingestion Pipeline - Offline data ingestion.

This package contains the data ingestion pipeline:
- Document loading
- Text splitting
- Transform (enrichment)
- Embedding
- Storage
"""

from src.ingestion.errors import classify_error
from src.ingestion.retry_policy import compute_backoff_delay
from src.ingestion.task_queue_backend import SQLiteTaskQueueBackend

__all__ = ["SQLiteTaskQueueBackend", "classify_error", "compute_backoff_delay"]
