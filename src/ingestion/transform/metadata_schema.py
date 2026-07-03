"""Schema helpers for structured metadata enrichment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


TITLE_MAX_CHARS = 60
SUMMARY_MAX_CHARS = 240
MAX_TAGS = 8
MAX_SECTION_DEPTH = 5
MAX_ENTITIES = 15
MAX_QUESTIONS = 5


@dataclass(frozen=True)
class MetadataEnrichmentConfig:
    """Runtime configuration for metadata enrichment."""

    enabled: bool = True
    use_llm: bool = False
    cache_enabled: bool = True
    cache_path: str = "data/cache/metadata_enrichment_cache.sqlite"
    max_tokens_per_chunk: int = 1200
    max_concurrency: int = 3
    budget_usd_per_run: float = 2.0
    output_schema: str = "json"
    fallback_to_rule_based: bool = True
    generate_questions: bool = True
    extract_entities: bool = True
    estimated_cost_per_1k_tokens: float = 0.0005
    prompt_path: str = "config/prompts/metadata_enrichment_json.txt"

    def cache_relevant_dict(self) -> dict[str, Any]:
        """Return stable config fields that should affect cache identity."""

        return {
            "use_llm": self.use_llm,
            "max_tokens_per_chunk": self.max_tokens_per_chunk,
            "output_schema": self.output_schema,
            "generate_questions": self.generate_questions,
            "extract_entities": self.extract_entities,
            "prompt_path": self.prompt_path,
        }


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _to_float(value: Any, default: float, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def build_metadata_enrichment_config(raw: dict[str, Any] | None) -> MetadataEnrichmentConfig:
    """Build config from dict while applying safe defaults."""

    raw = raw or {}
    default = MetadataEnrichmentConfig()
    return MetadataEnrichmentConfig(
        enabled=_to_bool(raw.get("enabled"), default.enabled),
        use_llm=_to_bool(raw.get("use_llm"), default.use_llm),
        cache_enabled=_to_bool(raw.get("cache_enabled"), default.cache_enabled),
        cache_path=str(raw.get("cache_path") or default.cache_path),
        max_tokens_per_chunk=_to_int(
            raw.get("max_tokens_per_chunk"),
            default.max_tokens_per_chunk,
        ),
        max_concurrency=_to_int(raw.get("max_concurrency"), default.max_concurrency),
        budget_usd_per_run=_to_float(
            raw.get("budget_usd_per_run"),
            default.budget_usd_per_run,
        ),
        output_schema=str(raw.get("output_schema") or default.output_schema),
        fallback_to_rule_based=_to_bool(
            raw.get("fallback_to_rule_based"),
            default.fallback_to_rule_based,
        ),
        generate_questions=_to_bool(
            raw.get("generate_questions"),
            default.generate_questions,
        ),
        extract_entities=_to_bool(raw.get("extract_entities"), default.extract_entities),
        estimated_cost_per_1k_tokens=_to_float(
            raw.get("estimated_cost_per_1k_tokens"),
            default.estimated_cost_per_1k_tokens,
        ),
        prompt_path=str(raw.get("prompt_path") or default.prompt_path),
    )


def truncate_string(value: Any, max_chars: int) -> str:
    """Normalise and truncate a string field."""

    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalise_string_list(
    value: Any,
    *,
    limit: int,
    max_item_chars: int = 80,
) -> list[str]:
    """Normalise list-like values to unique list[str]."""

    if value is None:
        items: list[Any] = []
    elif isinstance(value, str):
        items = re.split(r"[,，;；\n]+", value)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            raw = item.get("id") or item.get("text") or item.get("name") or ""
        else:
            raw = item
        text = truncate_string(raw, max_item_chars)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def normalise_page_range(value: Any) -> dict[str, int] | None:
    """Validate page_range as null or {start:int, end:int}."""

    if value is None or value == "":
        return None
    if isinstance(value, dict):
        start = value.get("start")
        end = value.get("end", start)
    elif isinstance(value, (list, tuple)) and value:
        start = value[0]
        end = value[-1]
    else:
        start = value
        end = value
    try:
        start_int = int(start)
        end_int = int(end)
    except (TypeError, ValueError):
        return None
    if start_int <= 0 or end_int <= 0:
        return None
    if end_int < start_int:
        start_int, end_int = end_int, start_int
    return {"start": start_int, "end": end_int}


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw LLM text or fenced Markdown."""

    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("empty LLM response")

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response does not contain JSON object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be an object")
    return parsed


def default_enrichment_metadata() -> dict[str, Any]:
    """Return defaults for all structured enrichment fields."""

    return {
        "title": "Untitled",
        "summary": "",
        "tags": [],
        "section_path": [],
        "heading_path": [],
        "page_range": None,
        "table_ids": [],
        "image_ids": [],
        "entities": [],
        "questions": [],
        "enrichment_method": "rule_based",
        "enrichment_cached": False,
    }


def add_text_companions(metadata: dict[str, Any]) -> dict[str, Any]:
    """Add Chroma-friendly text versions for nested/list fields."""

    output = dict(metadata)
    for key in (
        "tags",
        "section_path",
        "heading_path",
        "table_ids",
        "image_ids",
        "entities",
        "questions",
    ):
        values = output.get(key, [])
        if isinstance(values, list):
            separator = " > " if key.endswith("path") else ","
            output[f"{key}_text"] = separator.join(str(item) for item in values)
    page_range = output.get("page_range")
    if isinstance(page_range, dict):
        output["page_range_text"] = f"{page_range['start']}-{page_range['end']}"
    else:
        output["page_range_text"] = ""
    return output


def validate_enrichment_metadata(
    raw: dict[str, Any] | None,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and complete enrichment metadata.

    Unknown fields are preserved under ``raw_llm_metadata`` when they come from
    the LLM, while the top-level output remains a predictable schema.
    """

    fallback = fallback or {}
    raw = raw or {}
    result = default_enrichment_metadata()

    result["title"] = truncate_string(
        raw.get("title") or fallback.get("title") or result["title"],
        TITLE_MAX_CHARS,
    )
    result["summary"] = truncate_string(
        raw.get("summary") or fallback.get("summary") or result["summary"],
        SUMMARY_MAX_CHARS,
    )
    result["tags"] = normalise_string_list(
        raw.get("tags", fallback.get("tags")),
        limit=MAX_TAGS,
    )
    result["section_path"] = normalise_string_list(
        raw.get("section_path", fallback.get("section_path")),
        limit=MAX_SECTION_DEPTH,
    )
    result["heading_path"] = normalise_string_list(
        raw.get("heading_path", fallback.get("heading_path")),
        limit=MAX_SECTION_DEPTH,
    )
    result["page_range"] = normalise_page_range(
        raw.get("page_range", fallback.get("page_range"))
    )
    result["table_ids"] = normalise_string_list(
        raw.get("table_ids", fallback.get("table_ids")),
        limit=50,
    )
    result["image_ids"] = normalise_string_list(
        raw.get("image_ids", fallback.get("image_ids")),
        limit=50,
    )
    result["entities"] = normalise_string_list(
        raw.get("entities", fallback.get("entities")),
        limit=MAX_ENTITIES,
    )
    result["questions"] = normalise_string_list(
        raw.get("questions", fallback.get("questions")),
        limit=MAX_QUESTIONS,
        max_item_chars=140,
    )

    for key in ("enrichment_method", "fallback_reason"):
        value = raw.get(key, fallback.get(key))
        if value:
            result[key] = truncate_string(value, 120)
    if "enrichment_cached" in raw or "enrichment_cached" in fallback:
        result["enrichment_cached"] = bool(
            raw.get("enrichment_cached", fallback.get("enrichment_cached"))
        )

    extra = {
        key: value
        for key, value in raw.items()
        if key not in result and key not in {"raw_llm_metadata"}
    }
    if extra:
        result["raw_llm_metadata"] = json.dumps(extra, ensure_ascii=False, sort_keys=True)

    return add_text_companions(result)
