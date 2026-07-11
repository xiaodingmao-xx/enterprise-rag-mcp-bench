"""Small, traceable post-fusion score adjustments."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreBoostConfig:
    enabled: bool = False
    title_boost: float = 1.20
    heading_boost: float = 1.15
    tag_boost: float = 1.10
    exact_phrase_boost: float = 1.25
    source_type_boost: dict[str, float] = field(default_factory=dict)


class ScoreBooster:
    def __init__(self, config: ScoreBoostConfig | None = None) -> None:
        self.config = config or ScoreBoostConfig()

    def apply(self, query: str, results: list[Any]) -> list[Any]:
        if not self.config.enabled:
            return results
        terms = {item.lower() for item in re.findall(r"[\w\u4e00-\u9fff-]+", query) if item}
        for result in results:
            metadata = getattr(result, "metadata", {}) or {}
            score = float(getattr(result, "score", 0.0))
            reasons = []
            title = str(metadata.get("title", "")).lower()
            heading = " ".join(str(item) for item in metadata.get("heading_path", [])).lower()
            tags = " ".join(str(item) for item in metadata.get("tags", [])).lower()
            text = str(getattr(result, "text", "")).lower()
            if terms and any(term in title for term in terms):
                score *= self.config.title_boost
                reasons.append("title")
            if terms and any(term in heading for term in terms):
                score *= self.config.heading_boost
                reasons.append("heading")
            if terms and any(term in tags for term in terms):
                score *= self.config.tag_boost
                reasons.append("tag")
            if query.lower() in text or query.lower() in title or query.lower() in heading:
                score *= self.config.exact_phrase_boost
                reasons.append("exact_phrase")
            source_type = str(metadata.get("source_type", metadata.get("doc_type", ""))).lower()
            if source_type in self.config.source_type_boost:
                score *= self.config.source_type_boost[source_type]
                reasons.append(f"source_type:{source_type}")
            result.score = score
            if reasons:
                metadata["boost_reasons"] = reasons
                result.metadata = metadata
        return sorted(results, key=lambda item: item.score, reverse=True)
