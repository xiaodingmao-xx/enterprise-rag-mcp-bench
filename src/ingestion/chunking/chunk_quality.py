"""Rule-based chunk quality metrics and anomaly flags."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, Optional


def evaluate_chunk_quality(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    sibling_texts: Optional[Iterable[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = dict(config or {})
    value = str(text or "")
    characters = [char for char in value if not char.isspace()]
    sentences = [item for item in re.split(r"(?<=[。！？.!?])\s+|\n+", value.strip()) if item.strip()]
    punctuation = sum(unicodedata.category(char).startswith("P") for char in characters)
    suspicious = sum(
        char == "\ufffd" or unicodedata.category(char) in {"Cc", "Cf", "Co", "Cs"}
        for char in characters
    )
    garbled = suspicious / len(characters) if characters else 0.0
    duplicate = 0.0
    siblings = [str(item).strip().lower() for item in (sibling_texts or []) if str(item).strip()]
    if siblings:
        duplicate = 1.0 if value.strip().lower() in siblings else 0.0
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    line_counts = Counter(lines)
    repeated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
    duplicate = max(duplicate, repeated_lines / len(lines) if lines else 0.0)
    heading_count = len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+\S+", value))
    if metadata and metadata.get("heading"):
        heading_count = max(heading_count, 1)
    flags = []
    min_chars = int(config.get("min_chars", 50))
    max_chars = int(config.get("max_chars", 3000))
    if len(value.strip()) < min_chars:
        flags.append("too_short")
    if len(value) > max_chars:
        flags.append("too_long")
    if garbled > float(config.get("max_garbled_ratio", 0.2)):
        flags.append("garbled")
    if duplicate > float(config.get("max_duplicate_ratio", 0.4)):
        flags.append("duplicate")
    orphan = bool(metadata and metadata.get("heading") and len(value.strip()) <= max(min_chars, 80))
    if orphan:
        flags.append("orphan")
    table_lines = [line for line in value.splitlines() if "|" in line]
    if table_lines and (len(table_lines) < 3 or "---" not in "\n".join(table_lines[:3])):
        flags.append("incomplete_table")
    if value.count("```") % 2:
        flags.append("incomplete_code_block")
    boundary = 0.5
    if value.strip().endswith(("。", "！", "？", ".", "!", "?", ";", "；")):
        boundary += 0.25
    if value.lstrip().startswith(("#", "|", "```")):
        boundary += 0.1
    if "incomplete_table" in flags or "incomplete_code_block" in flags:
        boundary -= 0.25
    boundary = max(0.0, min(1.0, boundary))
    return {
        "character_count": len(value),
        "token_count": len(re.findall(r"[\w\u4e00-\u9fff]+", value, flags=re.UNICODE)),
        "sentence_count": len(sentences),
        "heading_count": heading_count,
        "duplicate_ratio": round(duplicate, 4),
        "garbled_ratio": round(garbled, 4),
        "punctuation_ratio": round(punctuation / len(characters) if characters else 0.0, 4),
        "orphan_heading": orphan,
        "semantic_boundary_score": round(boundary, 4),
        "flags": flags,
        "quality_status": "rejected" if config.get("reject_invalid") and flags else ("warning" if flags else "accepted"),
    }
