"""Error taxonomy used by reliable ingestion and task retry decisions."""

from __future__ import annotations

from typing import Any, Tuple


RETRYABLE_ERROR_CODES = {
    "PARSER_TIMEOUT",
    "EMBEDDING_TIMEOUT",
    "EMBEDDING_RATE_LIMIT",
    "VECTOR_STORE_TIMEOUT",
    "SPARSE_INDEX_TIMEOUT",
    "NETWORK_ERROR",
    "TEMPORARY_IO_ERROR",
    "UNKNOWN_TRANSIENT_ERROR",
}

NON_RETRYABLE_ERROR_CODES = {
    "UNSUPPORTED_FILE_TYPE",
    "FILE_HASH_MISMATCH",
    "EMPTY_DOCUMENT",
    "CORRUPTED_FILE",
    "INVALID_TENANT",
    "PERMISSION_DENIED",
    "INVALID_DOCUMENT_ID",
    "UNKNOWN_FATAL_ERROR",
}


class IngestionError(Exception):
    """Base exception carrying a stable error code."""

    code = "UNKNOWN_FATAL_ERROR"
    retryable = False

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


class RetryableIngestionError(IngestionError):
    retryable = True
    code = "UNKNOWN_TRANSIENT_ERROR"


class NonRetryableIngestionError(IngestionError):
    retryable = False


def classify_error(exc: BaseException) -> Tuple[str, bool]:
    """Map an exception to ``(error_code, retryable)``.

    Explicit ``IngestionError`` codes win.  The small heuristic fallback keeps
    ordinary network/timeout/IO exceptions useful to the task queue without
    coupling the queue to provider-specific exception classes.
    """

    code = getattr(exc, "code", None)
    retryable = getattr(exc, "retryable", None)
    if code:
        return str(code), bool(retryable)

    message = str(exc).lower()
    if "rate limit" in message or "429" in message:
        return "EMBEDDING_RATE_LIMIT", True
    if "timeout" in message or "timed out" in message:
        return "NETWORK_ERROR", True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return "TEMPORARY_IO_ERROR", True
    return "UNKNOWN_FATAL_ERROR", False

