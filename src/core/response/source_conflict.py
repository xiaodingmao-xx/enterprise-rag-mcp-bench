"""Deterministic source conflict warnings for retrieved contexts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class SourceConflictResult:
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return bool(self.conflicts)


class SourceConflictDetector:
    """Find obvious contradictory status/value statements without NLI."""

    STATUS_PAIRS = (
        ("已启用", "未启用"),
        ("已完成", "未完成"),
        ("通过", "未通过"),
        ("是", "否"),
        ("enabled", "disabled"),
        ("approved", "rejected"),
        ("completed", "incomplete"),
        ("passed", "failed"),
    )

    def detect(self, contexts: Iterable[Any]) -> SourceConflictResult:
        items = list(contexts)
        conflicts: list[dict[str, Any]] = []
        for left_index, left in enumerate(items):
            left_text = str(getattr(left, "text", "") or "")
            for right_index in range(left_index + 1, len(items)):
                right = items[right_index]
                right_text = str(getattr(right, "text", "") or "")
                pair = self._contradictory_pair(left_text, right_text)
                if pair:
                    conflicts.append(
                        {
                            "left_citation_id": getattr(left, "citation_id", f"C{left_index + 1}"),
                            "right_citation_id": getattr(right, "citation_id", f"C{right_index + 1}"),
                            "kind": "status",
                            "values": list(pair),
                        }
                    )
                field_conflict = self._field_value_conflict(left_text, right_text)
                if field_conflict:
                    conflicts.append(
                        {
                            "left_citation_id": getattr(left, "citation_id", f"C{left_index + 1}"),
                            "right_citation_id": getattr(right, "citation_id", f"C{right_index + 1}"),
                            "kind": "field_value",
                            "field": field_conflict,
                        }
                    )
        return SourceConflictResult(
            conflicts=conflicts,
            warnings=["CONFLICTING_SOURCES"] if conflicts else [],
        )

    def _contradictory_pair(self, left: str, right: str) -> tuple[str, str] | None:
        left_lower, right_lower = left.lower(), right.lower()
        for positive, negative in self.STATUS_PAIRS:
            if (positive.lower() in left_lower and negative.lower() in right_lower) or (
                negative.lower() in left_lower and positive.lower() in right_lower
            ):
                return positive, negative
        return None

    @staticmethod
    def _field_value_conflict(left: str, right: str) -> str | None:
        pattern = re.compile(r"([\w\u4e00-\u9fff]{2,20})\s*[:：=]\s*([^,，。；;\s]+)")
        left_fields = {match.group(1): match.group(2) for match in pattern.finditer(left)}
        right_fields = {match.group(1): match.group(2) for match in pattern.finditer(right)}
        for field_name, left_value in left_fields.items():
            right_value = right_fields.get(field_name)
            if right_value and right_value != left_value:
                return field_name
        return None

