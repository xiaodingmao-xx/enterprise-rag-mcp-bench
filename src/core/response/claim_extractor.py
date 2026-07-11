"""Rule-based claim extraction for citation verification."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class Claim:
    claim_id: str
    text: str
    citation_ids: list[str]
    sentence_index: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClaimExtractor(Protocol):
    def extract(self, answer: str) -> list[Claim]:
        ...


class RuleBasedClaimExtractor:
    """Extract factual-looking sentences without calling an LLM."""

    MARKER_RE = re.compile(r"\[C(\d+)\]", re.IGNORECASE)
    FACT_RE = re.compile(
        r"(?:\d|%|％|金额|元|美元|人民币|日期|版本|状态|结论|因此|说明|表明|显示|支持|需要|必须|已经|已|未|不能|可以|完成|启用|停用|通过|拒绝|approved|rejected|enabled|disabled|completed|failed)",
        re.IGNORECASE,
    )

    def __init__(self, max_claims: int = 20) -> None:
        self.max_claims = max(1, int(max_claims))

    def extract(self, answer: str) -> list[Claim]:
        claims: list[Claim] = []
        for sentence_index, raw_sentence in enumerate(self._split(answer or "")):
            sentence = " ".join(raw_sentence.split()).strip()
            if not sentence:
                continue
            citation_ids = [f"C{number}" for number in self.MARKER_RE.findall(sentence)]
            is_fact = bool(citation_ids or self.FACT_RE.search(sentence))
            if not is_fact:
                continue
            claim_id = f"CL{len(claims) + 1}"
            claims.append(
                Claim(
                    claim_id=claim_id,
                    text=sentence,
                    citation_ids=_dedupe(citation_ids),
                    sentence_index=sentence_index,
                    confidence=1.0 if citation_ids else 0.7,
                    metadata={"extractor": "rule_based"},
                )
            )
            if len(claims) >= self.max_claims:
                break
        return claims

    def _split(self, answer: str) -> list[str]:
        # Do not split decimal numbers such as ``2.0`` while supporting both
        # Chinese and English sentence punctuation.
        raw_parts = [part for part in re.split(r"(?<=[。！？!?；;])|(?<!\d)\.(?!\d)", answer) if part]
        parts: list[str] = []
        for part in raw_parts:
            if parts and part.strip().startswith("[") and self.MARKER_RE.search(part):
                parts[-1] += part
            else:
                parts.append(part)
        return parts


class NoopClaimExtractor:
    def extract(self, answer: str) -> list[Claim]:
        return []


class LLMClaimExtractor:
    """Reserved P2 interface; disabled by default."""

    def extract(self, answer: str) -> list[Claim]:
        raise NotImplementedError("LLM claim extraction is reserved for a future provider")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output
