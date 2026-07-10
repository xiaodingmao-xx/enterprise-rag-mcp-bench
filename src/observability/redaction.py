"""Security redaction primitives for logs, traces, and audit records.

The observability layer is deliberately conservative: values are redacted at
serialization time, so callers can continue to use the full document text
inside the ingestion/query pipeline without accidentally persisting it.
"""

from __future__ import annotations

import hashlib
import re
import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


REDACTED = "[REDACTED]"
TRUNCATED = "… [truncated]"

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?86[\s-]?)?1[3-9]\d{9}(?!\d)|\b\+?\d{3}[- ]\d{3}[- ]\d{4}\b"
)
_CN_ID_RE = re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_API_KEY_RE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|(?:AKIA|ASIA)[A-Z0-9]{12,}|AIza[A-Za-z0-9_-]{20,})"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|password|passwd|secret|access[_-]?token|client[_-]?secret|token)\b\s*[:=]\s*)([^\s,;]+)"
)
_USER_PATH_RE = re.compile(r"(?i)([\\/]Users[\\/])[^\\/]+")
_HOME_PATH_RE = re.compile(r"(?i)([\\/]home[\\/])[^\\/]+")

_TEXT_KEYS = {
    "text",
    "content",
    "page_content",
    "document_text",
    "chunk_text",
    "text_preview",
    "text_before",
    "text_after",
    "original_query",
    "query",
    "answer",
    "generated_answer",
}
_CHUNK_TEXT_KEYS = {
    "text",
    "content",
    "page_content",
    "document_text",
    "chunk_text",
    "text_before",
    "text_after",
}
_PROMPT_KEYS = {"prompt", "full_prompt", "system_prompt", "user_prompt", "messages"}
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "password",
    "passwd",
    "secret",
    "client_secret",
}
_PATH_KEYS = {"source_path", "stored_path", "file_path", "path", "filename"}


@dataclass(frozen=True)
class RedactionConfig:
    """Runtime redaction policy.

    ``enabled=False`` and ``include_prompt=True`` are intentionally restricted
    to development environments.  This prevents a production deployment from
    disabling the only persistence-time safety boundary by configuration error.
    """

    enabled: bool = True
    max_text_length: int = 256
    hash_user_id: bool = True
    include_chunk_text: bool = False
    include_prompt: bool = False
    sampling_rate: float = 1.0
    environment: str = "development"

    def __post_init__(self) -> None:
        environment = str(self.environment or "development").strip().lower()
        object.__setattr__(self, "environment", environment)
        if self.max_text_length < 1:
            raise ValueError("observability.redaction.max_text_length must be positive")
        if not 0.0 <= float(self.sampling_rate) <= 1.0:
            raise ValueError("observability.trace.sampling_rate must be between 0 and 1")
        if environment in {"prod", "production"} and not self.enabled:
            raise ValueError("observability.redaction.enabled=false is allowed only in development")
        if environment in {"prod", "production"} and self.include_prompt:
            raise ValueError("observability.trace.include_prompt=true is forbidden in production")


def hash_user_id(user_id: Any, *, enabled: bool = True) -> str:
    """Return a stable non-reversible identifier for an actor."""

    if user_id is None or str(user_id).strip() == "":
        return "anonymous"
    if not enabled:
        return "user-redacted"
    digest = hashlib.sha256(str(user_id).strip().encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def redact_text(value: Any, *, max_length: int = 256, is_path: bool = False) -> str:
    """Mask common PII/secrets and truncate document-like text."""

    text = "" if value is None else str(value)
    text = _CN_ID_RE.sub(REDACTED, text)
    text = _BEARER_RE.sub("Bearer " + REDACTED, text)
    text = _API_KEY_RE.sub(REDACTED, text)
    text = _SECRET_ASSIGNMENT_RE.sub(lambda match: match.group(1) + REDACTED, text)
    text = _EMAIL_RE.sub(REDACTED, text)
    text = _PHONE_RE.sub(REDACTED, text)
    if is_path:
        text = _USER_PATH_RE.sub(r"\1[USER]", text)
        text = _HOME_PATH_RE.sub(r"\1[USER]", text)
    if len(text) > max_length:
        return text[:max_length] + TRUNCATED
    return text


def redact_exception(value: Any, *, max_length: int = 512) -> str:
    """Create a safe, bounded exception string for ordinary logs."""

    return redact_text(value, max_length=max_length)


def _safe_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_")


def _preview_key(key: str) -> str:
    if key == "text_before":
        return "text_before_preview"
    if key == "text_after":
        return "text_after_preview"
    if key in _CHUNK_TEXT_KEYS or key == "text_preview":
        return "redacted_preview"
    if key in _PROMPT_KEYS:
        return "prompt_redacted"
    return "query_preview"


def redact_value(value: Any, config: Optional[RedactionConfig] = None, *, key: str = "") -> Any:
    """Recursively redact arbitrary JSON-compatible data."""

    cfg = config or RedactionConfig()
    normalized_key = _safe_key(key)

    if normalized_key in _SECRET_KEYS:
        return REDACTED
    if isinstance(value, Mapping):
        return redact_mapping(value, cfg)
    if isinstance(value, list):
        return [redact_value(item, cfg, key=key) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item, cfg, key=key) for item in value]
    if isinstance(value, str):
        return redact_text(value, max_length=cfg.max_text_length, is_path=normalized_key in _PATH_KEYS)
    return value


def redact_mapping(value: Mapping[str, Any], config: Optional[RedactionConfig] = None) -> Dict[str, Any]:
    """Recursively redact a mapping while keeping operational identifiers."""

    cfg = config or RedactionConfig()
    output: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        normalized_key = _safe_key(key)

        if normalized_key in _SECRET_KEYS:
            output[key] = REDACTED
            continue

        if normalized_key in _PROMPT_KEYS and not cfg.include_prompt:
            output[_preview_key(normalized_key)] = REDACTED
            continue

        if normalized_key in _TEXT_KEYS:
            preview = redact_text(
                raw_value,
                max_length=cfg.max_text_length,
                is_path=normalized_key in _PATH_KEYS,
            )
            if normalized_key in _CHUNK_TEXT_KEYS and not cfg.include_chunk_text:
                output[_preview_key(normalized_key)] = preview
            elif normalized_key in _PROMPT_KEYS and not cfg.include_prompt:
                output[_preview_key(normalized_key)] = REDACTED
            elif normalized_key in {"query", "original_query", "answer", "generated_answer"}:
                output[_preview_key(normalized_key)] = preview
            else:
                output[key] = preview
            continue

        if isinstance(raw_value, Mapping):
            output[key] = redact_mapping(raw_value, cfg)
        elif isinstance(raw_value, list):
            output[key] = [
                redact_mapping(item, cfg) if isinstance(item, Mapping) else redact_value(item, cfg, key=key)
                for item in raw_value
            ]
        else:
            output[key] = redact_value(raw_value, cfg, key=key)
    return output


def redact_trace_payload(payload: Mapping[str, Any], config: Optional[RedactionConfig] = None) -> Dict[str, Any]:
    """Redact a trace payload and guarantee no raw user identifier is emitted."""

    cfg = config or RedactionConfig()
    if not cfg.enabled:
        return copy.deepcopy(dict(payload))
    result = redact_mapping(payload, cfg)
    if "user_id" in result:
        result.pop("user_id", None)
    return result


def config_from_settings(settings: Any = None) -> RedactionConfig:
    """Build a policy from Settings, a mapping, or safe defaults."""

    if isinstance(settings, RedactionConfig):
        return settings

    observability = getattr(settings, "observability", None) if settings is not None else None
    if observability is None and isinstance(settings, Mapping):
        observability = settings.get("observability", settings)
    if observability is None:
        return RedactionConfig()

    def read(obj: Any, name: str, default: Any) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(name, default)
        return getattr(obj, name, default)

    redaction = read(observability, "redaction", {})
    trace = read(observability, "trace", {})
    if not isinstance(redaction, (Mapping, RedactionConfig)) and not hasattr(redaction, "enabled"):
        redaction = {}
    if not isinstance(trace, Mapping) and not hasattr(trace, "include_chunk_text"):
        trace = {}

    def nested(obj: Any, name: str, default: Any) -> Any:
        if isinstance(obj, RedactionConfig):
            return getattr(obj, name, default)
        if isinstance(obj, Mapping):
            return obj.get(name, default)
        return getattr(obj, name, default)

    environment = read(observability, "environment", None)
    if not environment:
        environment = "development"
    return RedactionConfig(
        enabled=bool(nested(redaction, "enabled", True)),
        max_text_length=int(nested(redaction, "max_text_length", 256)),
        hash_user_id=bool(nested(redaction, "hash_user_id", True)),
        include_chunk_text=bool(nested(trace, "include_chunk_text", False)),
        include_prompt=bool(nested(trace, "include_prompt", False)),
        sampling_rate=float(nested(trace, "sampling_rate", 1.0)),
        environment=str(environment),
    )
