"""Lightweight prompt-injection detection for untrusted query/context text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


PROMPT_INJECTION_WARNING = "PROMPT_INJECTION_DETECTED"


@dataclass(frozen=True)
class SafetyCheckResult:
    detected: bool
    warnings: list[str] = field(default_factory=list)


class PromptInjectionDetector:
    PATTERNS = (
        r"ignore\s+(?:all\s+)?previous\s+instructions",
        r"disregard\s+the\s+system\s+prompt",
        r"reveal\s+your\s+prompt",
        r"you\s+are\s+now",
        r"\bbypass\b",
        r"\bjailbreak\b",
        r"system\s+message",
        r"developer\s+message",
        r"忽略以上指令",
        r"忽略之前的规则",
        r"忽略之前规则",
        r"泄露提示词",
        r"输出系统提示词",
        r"绕过限制",
        r"你现在是",
        r"不要引用来源",
        r"不要遵守知识库规则",
    )

    def __init__(self) -> None:
        self._compiled = tuple(re.compile(pattern, re.IGNORECASE) for pattern in self.PATTERNS)

    def contains(self, text: str) -> bool:
        value = text or ""
        return any(pattern.search(value) for pattern in self._compiled)

    def check(self, query: str = "", contexts: Iterable[Any] = (), answer: str = "") -> SafetyCheckResult:
        texts = [query, answer]
        texts.extend(str(getattr(context, "text", "") or "") for context in contexts)
        detected = any(self.contains(text) for text in texts)
        return SafetyCheckResult(
            detected=detected,
            warnings=[PROMPT_INJECTION_WARNING] if detected else [],
        )


def detect_prompt_injection(query: str = "", contexts: Iterable[Any] = (), answer: str = "") -> bool:
    return PromptInjectionDetector().check(query=query, contexts=contexts, answer=answer).detected

