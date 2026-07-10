"""Configuration loading and validation for the Modular RAG MCP Server."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# ---------------------------------------------------------------------------
# Repo root & path resolution
# ---------------------------------------------------------------------------
# Anchored to this file's location: <repo>/src/core/settings.py → parents[2]
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Default absolute path to settings.yaml
DEFAULT_SETTINGS_PATH: Path = REPO_ROOT / "config" / "settings.yaml"

DEFAULT_SUPPORTED_EXTENSIONS: List[str] = [
    ".pdf",
    ".md",
    ".txt",
    ".html",
    ".htm",
    ".docx",
    ".py",
    ".js",
    ".java",
]

_ENV_PLACEHOLDER_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
)


def _expand_env_placeholders(raw_yaml: str) -> str:
    """Expand ${ENV_NAME} and ${ENV_NAME:-default} placeholders in YAML text."""

    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(env_name)

        if default is not None:
            return env_value if env_value else default
        return env_value if env_value is not None else ""

    return _ENV_PLACEHOLDER_PATTERN.sub(replace, raw_yaml)


def resolve_path(relative: Union[str, Path]) -> Path:
    """Resolve a repo-relative path to an absolute path.

    If *relative* is already absolute it is returned as-is.  Otherwise
    it is resolved against :data:`REPO_ROOT`.

    >>> resolve_path("config/settings.yaml")  # doctest: +SKIP
    PosixPath('/home/user/Modular-RAG-MCP-Server/config/settings.yaml')
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


class SettingsError(ValueError):
    """Raised when settings validation fails."""


def _require_mapping(data: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    value = data.get(key)
    if value is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    if not isinstance(value, dict):
        raise SettingsError(f"Expected mapping for field: {path}.{key}")
    return value


def _require_value(data: Dict[str, Any], key: str, path: str) -> Any:
    if key not in data or data.get(key) is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    return data[key]


def _require_str(data: Dict[str, Any], key: str, path: str) -> str:
    value = _require_value(data, key, path)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Expected non-empty string for field: {path}.{key}")
    return value


def _require_int(data: Dict[str, Any], key: str, path: str) -> int:
    value = _require_value(data, key, path)
    if not isinstance(value, int):
        raise SettingsError(f"Expected integer for field: {path}.{key}")
    return value


def _require_number(data: Dict[str, Any], key: str, path: str) -> float:
    value = _require_value(data, key, path)
    if not isinstance(value, (int, float)):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    return float(value)


def _require_bool(data: Dict[str, Any], key: str, path: str) -> bool:
    value = _require_value(data, key, path)
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    return value


def _require_list(data: Dict[str, Any], key: str, path: str) -> List[Any]:
    value = _require_value(data, key, path)
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    return value


def _optional_positive_int(data: Dict[str, Any], key: str, path: str) -> Optional[int]:
    if key not in data or data.get(key) is None:
        return None
    value = data.get(key)
    if not isinstance(value, int) or value <= 0:
        raise SettingsError(f"Expected positive integer for field: {path}.{key}")
    return value


def _normalise_extensions(value: Any, path: str) -> List[str]:
    if value is None:
        return DEFAULT_SUPPORTED_EXTENSIONS.copy()
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}")

    extensions: List[str] = []
    for item in value:
        extension = str(item).strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension not in extensions:
            extensions.append(extension)

    return extensions or DEFAULT_SUPPORTED_EXTENSIONS.copy()


def _optional_mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_or_default(value: Any, default: int, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _float_or_default(value: Any, default: float, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    extra_body: Optional[Dict[str, Any]] = None
    # Azure/OpenAI-specific optional fields
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama-specific optional fields
    base_url: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int
    max_batch_size: Optional[int] = None
    # Azure-specific optional fields
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama-specific optional fields
    base_url: Optional[str] = None


@dataclass(frozen=True)
class VectorStoreSettings:
    provider: str
    persist_directory: str
    collection_name: str


@dataclass(frozen=True)
class FTS5Settings:
    db_path: str = "./data/db/sparse_fts5.db"
    tokenizer: str = "unicode61"
    match_mode: str = "or"
    busy_timeout_ms: int = 10000
    max_retries: int = 3


@dataclass(frozen=True)
class RetrievalSettings:
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int
    sparse_backend: str = "json_bm25"
    fts5: FTS5Settings = field(default_factory=FTS5Settings)


@dataclass(frozen=True)
class RerankSettings:
    enabled: bool
    provider: str
    model: str
    top_k: int
    candidate_top_k: int = 20
    output_top_k: int = 5
    timeout_seconds: float = 30.0
    fallback_on_timeout: bool = True
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    api_format: str = "dashscope"
    instruct: Optional[str] = None
    return_documents: bool = True


@dataclass(frozen=True)
class EvaluationSettings:
    enabled: bool
    provider: str
    metrics: List[str]


@dataclass(frozen=True)
class RedactionSettings:
    enabled: bool = True
    max_text_length: int = 256
    hash_user_id: bool = True


@dataclass(frozen=True)
class TraceSettings:
    include_chunk_text: bool = False
    include_prompt: bool = False
    sampling_rate: float = 1.0


@dataclass(frozen=True)
class RetentionSettings:
    max_days: int = 30
    max_file_size_mb: int = 100
    rotation_count: int = 5


@dataclass(frozen=True)
class JWTSettings:
    issuer: str = ""
    audience: str = ""
    secret: str = ""


@dataclass(frozen=True)
class ACLSettings:
    enabled: bool = True


@dataclass(frozen=True)
class SecuritySettings:
    enabled: bool = True
    mode: str = "local-dev"
    require_tenant: bool = False
    require_authentication: bool = False
    default_local_tenant: str = "local"
    default_local_user: str = "local-user"
    jwt: JWTSettings = field(default_factory=JWTSettings)
    acl: ACLSettings = field(default_factory=ACLSettings)


@dataclass(frozen=True)
class ObservabilitySettings:
    log_level: str
    trace_enabled: bool
    trace_file: str
    structured_logging: bool
    environment: str = "development"
    audit_file: str = "./logs/audit.jsonl"
    operational_file: str = "./logs/operational.jsonl"
    redaction: RedactionSettings = field(default_factory=RedactionSettings)
    trace: TraceSettings = field(default_factory=TraceSettings)
    retention: RetentionSettings = field(default_factory=RetentionSettings)


@dataclass(frozen=True)
class VisionLLMSettings:
    enabled: bool
    provider: str
    model: str
    max_image_size: int
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    base_url: Optional[str] = None


@dataclass(frozen=True)
class ChunkingSettings:
    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    recursive: Dict[str, Any] = field(default_factory=dict)
    markdown_header: Dict[str, Any] = field(default_factory=dict)
    semantic: Dict[str, Any] = field(default_factory=dict)
    parent_child: Dict[str, Any] = field(default_factory=dict)
    sliding_window: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionSettings:
    chunk_size: int
    chunk_overlap: int
    splitter: str
    batch_size: int
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    chunk_refiner: Optional[Dict[str, Any]] = None  # 动态配置
    metadata_enricher: Optional[Dict[str, Any]] = None  # 动态配置
    metadata_enrichment: Optional[Dict[str, Any]] = None  # 结构化元数据增强配置
    document_quality: Optional[Dict[str, Any]] = None
    concurrent_upload: Dict[str, Any] = field(default_factory=dict)
    task_queue: Dict[str, Any] = field(default_factory=dict)
    supported_extensions: List[str] = field(
        default_factory=lambda: DEFAULT_SUPPORTED_EXTENSIONS.copy()
    )


@dataclass(frozen=True)
class QueryCacheSettings:
    enabled: bool = True
    max_size: int = 128
    ttl_seconds: int = 300


@dataclass(frozen=True)
class EmbeddingCacheSettings:
    enabled: bool = True
    db_path: str = "./data/db/embedding_cache.db"


@dataclass(frozen=True)
class HallucinationGuardSettings:
    enabled: bool = True


@dataclass(frozen=True)
class AnswerGenerationSettings:
    enabled: bool = True
    default_mode: str = "contexts"
    min_contexts: int = 1
    min_score: float = 0.2
    max_context_chars: int = 8000
    default_answer_style: str = "concise"
    timeout_seconds: float = 20.0
    hallucination_guard: HallucinationGuardSettings = field(
        default_factory=HallucinationGuardSettings
    )


@dataclass(frozen=True)
class ResponseSettings:
    answer_generation: AnswerGenerationSettings = field(
        default_factory=AnswerGenerationSettings
    )


@dataclass(frozen=True)
class PerformanceSettings:
    query_cache: QueryCacheSettings = field(default_factory=QueryCacheSettings)
    embedding_cache: EmbeddingCacheSettings = field(default_factory=EmbeddingCacheSettings)


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    security: SecuritySettings = field(default_factory=SecuritySettings)
    ingestion: Optional[IngestionSettings] = None
    vision_llm: Optional[VisionLLMSettings] = None
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    response: ResponseSettings = field(default_factory=ResponseSettings)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        if not isinstance(data, dict):
            raise SettingsError("Settings root must be a mapping")

        llm = _require_mapping(data, "llm", "settings")
        embedding = _require_mapping(data, "embedding", "settings")
        vector_store = _require_mapping(data, "vector_store", "settings")
        retrieval = _require_mapping(data, "retrieval", "settings")
        rerank = _require_mapping(data, "rerank", "settings")
        evaluation = _require_mapping(data, "evaluation", "settings")
        observability = _require_mapping(data, "observability", "settings")

        ingestion_settings = None
        if "ingestion" in data:
            ingestion = _require_mapping(data, "ingestion", "settings")
            legacy_chunk_size = _require_int(ingestion, "chunk_size", "ingestion")
            legacy_chunk_overlap = _require_int(ingestion, "chunk_overlap", "ingestion")
            legacy_splitter = _require_str(ingestion, "splitter", "ingestion")
            chunking = _optional_mapping(ingestion.get("chunking"))
            chunking_settings = ChunkingSettings(
                strategy=str(chunking.get("strategy") or legacy_splitter or "recursive"),
                chunk_size=_int_or_default(chunking.get("chunk_size"), legacy_chunk_size, 1),
                chunk_overlap=_int_or_default(
                    chunking.get("chunk_overlap"),
                    legacy_chunk_overlap,
                    0,
                ),
                recursive=_optional_mapping(chunking.get("recursive")),
                markdown_header=_optional_mapping(chunking.get("markdown_header")),
                semantic=_optional_mapping(chunking.get("semantic")),
                parent_child=_optional_mapping(chunking.get("parent_child")),
                sliding_window=_optional_mapping(chunking.get("sliding_window")),
            )
            ingestion_settings = IngestionSettings(
                chunk_size=legacy_chunk_size,
                chunk_overlap=legacy_chunk_overlap,
                splitter=legacy_splitter,
                batch_size=_require_int(ingestion, "batch_size", "ingestion"),
                chunking=chunking_settings,
                chunk_refiner=ingestion.get("chunk_refiner"),  # 可选配置
                metadata_enricher=ingestion.get("metadata_enricher"),  # 可选配置
                metadata_enrichment=ingestion.get("metadata_enrichment"),  # 可选配置
                document_quality=ingestion.get("document_quality"),
                supported_extensions=_normalise_extensions(
                    ingestion.get("supported_extensions"),
                    "ingestion.supported_extensions",
                ),
            )

        if ingestion_settings is not None:
            object.__setattr__(
                ingestion_settings,
                "metadata_enricher",
                ingestion.get("metadata_enricher"),
            )
            object.__setattr__(
                ingestion_settings,
                "metadata_enrichment",
                ingestion.get("metadata_enrichment"),
            )
            object.__setattr__(
                ingestion_settings,
                "document_quality",
                ingestion.get("document_quality"),
            )
            object.__setattr__(
                ingestion_settings,
                "concurrent_upload",
                _optional_mapping(ingestion.get("concurrent_upload")),
            )
            object.__setattr__(
                ingestion_settings,
                "task_queue",
                _optional_mapping(ingestion.get("task_queue")),
            )

        vision_llm_settings = None
        if "vision_llm" in data:
            vision_llm = _require_mapping(data, "vision_llm", "settings")
            vision_llm_settings = VisionLLMSettings(
                enabled=_require_bool(vision_llm, "enabled", "vision_llm"),
                provider=_require_str(vision_llm, "provider", "vision_llm"),
                model=_require_str(vision_llm, "model", "vision_llm"),
                max_image_size=_require_int(vision_llm, "max_image_size", "vision_llm"),
                api_key=vision_llm.get("api_key"),
                api_version=vision_llm.get("api_version"),
                azure_endpoint=vision_llm.get("azure_endpoint"),
                deployment_name=vision_llm.get("deployment_name"),
                base_url=vision_llm.get("base_url"),
            )

        performance_settings = PerformanceSettings()
        if "performance" in data:
            performance = _require_mapping(data, "performance", "settings")

            query_cache_data = performance.get("query_cache") or {}
            if not isinstance(query_cache_data, dict):
                raise SettingsError("Expected mapping for field: performance.query_cache")

            embedding_cache_data = performance.get("embedding_cache") or {}
            if not isinstance(embedding_cache_data, dict):
                raise SettingsError("Expected mapping for field: performance.embedding_cache")

            performance_settings = PerformanceSettings(
                query_cache=QueryCacheSettings(
                    enabled=bool(query_cache_data.get("enabled", True)),
                    max_size=int(query_cache_data.get("max_size", 128)),
                    ttl_seconds=int(query_cache_data.get("ttl_seconds", 300)),
                ),
                embedding_cache=EmbeddingCacheSettings(
                    enabled=bool(embedding_cache_data.get("enabled", True)),
                    db_path=str(
                        embedding_cache_data.get(
                            "db_path",
                            "./data/db/embedding_cache.db",
                        )
                    ),
                ),
            )

        response_settings = ResponseSettings()
        if "response" in data:
            response = _require_mapping(data, "response", "settings")
            answer_generation = _optional_mapping(response.get("answer_generation"))
            guard_settings = _optional_mapping(answer_generation.get("hallucination_guard"))
            response_settings = ResponseSettings(
                answer_generation=AnswerGenerationSettings(
                    enabled=bool(answer_generation.get("enabled", True)),
                    default_mode=str(answer_generation.get("default_mode", "contexts")),
                    min_contexts=_int_or_default(
                        answer_generation.get("min_contexts"),
                        1,
                        1,
                    ),
                    min_score=_float_or_default(
                        answer_generation.get("min_score"),
                        0.2,
                        0.0,
                    ),
                    max_context_chars=_int_or_default(
                        answer_generation.get("max_context_chars"),
                        8000,
                        1,
                    ),
                    default_answer_style=str(
                        answer_generation.get("default_answer_style", "concise")
                    ),
                    timeout_seconds=_float_or_default(
                        answer_generation.get("timeout_seconds"),
                        20.0,
                        0.001,
                    ),
                    hallucination_guard=HallucinationGuardSettings(
                        enabled=bool(guard_settings.get("enabled", True))
                    ),
                )
            )

        rerank_top_k = _require_int(rerank, "top_k", "rerank")

        observability_redaction = _optional_mapping(observability.get("redaction"))
        observability_trace = _optional_mapping(observability.get("trace"))
        observability_retention = _optional_mapping(observability.get("retention"))
        observability_environment = str(
            observability.get(
                "environment",
                os.environ.get("APP_ENV", os.environ.get("ENVIRONMENT", "development")),
            )
            or "development"
        ).strip().lower()
        observability_settings = ObservabilitySettings(
            log_level=_require_str(observability, "log_level", "observability"),
            trace_enabled=_require_bool(observability, "trace_enabled", "observability"),
            trace_file=_require_str(observability, "trace_file", "observability"),
            structured_logging=_require_bool(observability, "structured_logging", "observability"),
            environment=observability_environment,
            audit_file=str(observability.get("audit_file", "./logs/audit.jsonl")),
            operational_file=str(observability.get("operational_file", "./logs/operational.jsonl")),
            redaction=RedactionSettings(
                enabled=bool(observability_redaction.get("enabled", True)),
                max_text_length=_int_or_default(observability_redaction.get("max_text_length"), 256, 1),
                hash_user_id=bool(observability_redaction.get("hash_user_id", True)),
            ),
            trace=TraceSettings(
                include_chunk_text=bool(observability_trace.get("include_chunk_text", False)),
                include_prompt=bool(observability_trace.get("include_prompt", False)),
                sampling_rate=_float_or_default(observability_trace.get("sampling_rate"), 1.0, 0.0),
            ),
            retention=RetentionSettings(
                max_days=_int_or_default(observability_retention.get("max_days"), 30, 0),
                max_file_size_mb=_int_or_default(observability_retention.get("max_file_size_mb"), 100, 1),
                rotation_count=_int_or_default(observability_retention.get("rotation_count"), 5, 1),
            ),
        )
        # Force policy validation even when settings are loaded from a minimal
        # test configuration.
        from src.observability.redaction import RedactionConfig

        RedactionConfig(
            enabled=observability_settings.redaction.enabled,
            max_text_length=observability_settings.redaction.max_text_length,
            hash_user_id=observability_settings.redaction.hash_user_id,
            include_chunk_text=observability_settings.trace.include_chunk_text,
            include_prompt=observability_settings.trace.include_prompt,
            sampling_rate=observability_settings.trace.sampling_rate,
            environment=observability_settings.environment,
        )

        security_data = _optional_mapping(data.get("security"))
        security_jwt = _optional_mapping(security_data.get("jwt"))
        security_acl = _optional_mapping(security_data.get("acl"))
        security_settings = SecuritySettings(
            enabled=bool(security_data.get("enabled", True)),
            mode=str(security_data.get("mode", "local-dev")).strip().lower(),
            require_tenant=bool(security_data.get("require_tenant", False)),
            require_authentication=bool(security_data.get("require_authentication", False)),
            default_local_tenant=str(security_data.get("default_local_tenant", "local")),
            default_local_user=str(security_data.get("default_local_user", "local-user")),
            jwt=JWTSettings(
                issuer=str(security_jwt.get("issuer", "")),
                audience=str(security_jwt.get("audience", "")),
                secret=str(security_jwt.get("secret", os.environ.get("JWT_SECRET", ""))),
            ),
            acl=ACLSettings(enabled=bool(security_acl.get("enabled", True))),
        )

        settings = cls(
            llm=LLMSettings(
                provider=_require_str(llm, "provider", "llm"),
                model=_require_str(llm, "model", "llm"),
                temperature=_require_number(llm, "temperature", "llm"),
                max_tokens=_require_int(llm, "max_tokens", "llm"),
                extra_body=llm.get("extra_body") if isinstance(llm.get("extra_body"), dict) else None,
                api_key=llm.get("api_key"),
                api_version=llm.get("api_version"),
                azure_endpoint=llm.get("azure_endpoint"),
                deployment_name=llm.get("deployment_name"),
                base_url=llm.get("base_url"),
            ),
            embedding=EmbeddingSettings(
                provider=_require_str(embedding, "provider", "embedding"),
                model=_require_str(embedding, "model", "embedding"),
                dimensions=_require_int(embedding, "dimensions", "embedding"),
                max_batch_size=_optional_positive_int(
                    embedding,
                    "max_batch_size",
                    "embedding",
                ),
                api_key=embedding.get("api_key"),
                api_version=embedding.get("api_version"),
                azure_endpoint=embedding.get("azure_endpoint"),
                deployment_name=embedding.get("deployment_name"),
                base_url=embedding.get("base_url"),
            ),
            vector_store=VectorStoreSettings(
                provider=_require_str(vector_store, "provider", "vector_store"),
                persist_directory=_require_str(vector_store, "persist_directory", "vector_store"),
                collection_name=_require_str(vector_store, "collection_name", "vector_store"),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=_require_int(retrieval, "dense_top_k", "retrieval"),
                sparse_top_k=_require_int(retrieval, "sparse_top_k", "retrieval"),
                fusion_top_k=_require_int(retrieval, "fusion_top_k", "retrieval"),
                rrf_k=_require_int(retrieval, "rrf_k", "retrieval"),
                sparse_backend=str(retrieval.get("sparse_backend", "json_bm25")),
                fts5=FTS5Settings(
                    db_path=str(
                        _optional_mapping(retrieval.get("fts5")).get(
                            "db_path",
                            "./data/db/sparse_fts5.db",
                        )
                    ),
                    tokenizer=str(
                        _optional_mapping(retrieval.get("fts5")).get(
                            "tokenizer",
                            "unicode61",
                        )
                    ),
                    match_mode=str(
                        _optional_mapping(retrieval.get("fts5")).get(
                            "match_mode",
                            "or",
                        )
                    ),
                    busy_timeout_ms=_int_or_default(
                        _optional_mapping(retrieval.get("fts5")).get("busy_timeout_ms"),
                        10000,
                        1,
                    ),
                    max_retries=_int_or_default(
                        _optional_mapping(retrieval.get("fts5")).get("max_retries"),
                        3,
                        0,
                    ),
                ),
            ),
            rerank=RerankSettings(
                enabled=_require_bool(rerank, "enabled", "rerank"),
                provider=_require_str(rerank, "provider", "rerank"),
                model=_require_str(rerank, "model", "rerank"),
                top_k=rerank_top_k,
                candidate_top_k=_int_or_default(
                    rerank.get("candidate_top_k"),
                    max(rerank_top_k * 2, rerank_top_k),
                    1,
                ),
                output_top_k=_int_or_default(
                    rerank.get("output_top_k"),
                    rerank_top_k,
                    1,
                ),
                timeout_seconds=_float_or_default(
                    rerank.get("timeout_seconds", rerank.get("timeout")),
                    30.0,
                    0.001,
                ),
                fallback_on_timeout=bool(
                    rerank.get(
                        "fallback_on_timeout",
                        rerank.get("fallback_on_error", True),
                    )
                ),
                api_key=(
                    str(rerank.get("api_key")).strip()
                    if rerank.get("api_key") is not None
                    and str(rerank.get("api_key")).strip()
                    else None
                ),
                endpoint=(
                    str(rerank.get("endpoint")).strip()
                    if rerank.get("endpoint") is not None
                    and str(rerank.get("endpoint")).strip()
                    else None
                ),
                api_format=(
                    str(rerank.get("api_format", "dashscope")).strip().lower()
                    or "dashscope"
                ),
                instruct=(
                    str(rerank.get("instruct")).strip()
                    if rerank.get("instruct") is not None
                    and str(rerank.get("instruct")).strip()
                    else None
                ),
                return_documents=bool(rerank.get("return_documents", True)),
            ),
            evaluation=EvaluationSettings(
                enabled=_require_bool(evaluation, "enabled", "evaluation"),
                provider=_require_str(evaluation, "provider", "evaluation"),
                metrics=[str(item) for item in _require_list(evaluation, "metrics", "evaluation")],
            ),
            observability=observability_settings,
            security=security_settings,
            ingestion=ingestion_settings,
            vision_llm=vision_llm_settings,
            performance=performance_settings,
            response=response_settings,
        )

        return settings


def validate_settings(settings: Settings) -> None:
    """Validate settings and raise SettingsError if invalid."""

    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    if not settings.vector_store.provider:
        raise SettingsError("Missing required field: vector_store.provider")
    if not settings.retrieval.rrf_k:
        raise SettingsError("Missing required field: retrieval.rrf_k")
    if not settings.rerank.provider:
        raise SettingsError("Missing required field: rerank.provider")
    if not settings.evaluation.provider:
        raise SettingsError("Missing required field: evaluation.provider")
    if not settings.observability.log_level:
        raise SettingsError("Missing required field: observability.log_level")
    if settings.security.mode not in {"local-dev", "development", "production", "prod"}:
        raise SettingsError("security.mode must be local-dev or production")
    if settings.security.mode in {"production", "prod"}:
        if not settings.security.enabled:
            raise SettingsError("security.enabled=false is forbidden in production")
        if not settings.security.require_tenant:
            raise SettingsError("production mode requires security.require_tenant=true")
        if not settings.security.require_authentication:
            raise SettingsError("production mode requires security.require_authentication=true")
        if not settings.observability.redaction.enabled:
            raise SettingsError("production mode requires observability.redaction.enabled=true")
        if settings.observability.trace.include_prompt:
            raise SettingsError("production mode forbids observability.trace.include_prompt=true")


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a YAML file and validate required fields.

    Args:
        path: Path to settings YAML.  Defaults to
            ``<repo>/config/settings.yaml`` (absolute, CWD-independent).
    """
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_absolute():
        settings_path = resolve_path(settings_path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as handle:
        raw_yaml = _expand_env_placeholders(handle.read())
        data = yaml.safe_load(raw_yaml)

    settings = Settings.from_dict(data or {})
    validate_settings(settings)
    return settings
