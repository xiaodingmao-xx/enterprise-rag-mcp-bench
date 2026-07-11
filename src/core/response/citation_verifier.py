"""Rule-based claim-level citation verification with ACL enforcement."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

from src.core.response.citation import CitationRecord
from src.core.response.claim_extractor import Claim
from src.security.acl_filter import ACLFilter
from src.security.context import RequestContext


class CitationVerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    INVALID_CITATION = "invalid_citation"


@dataclass
class ClaimVerification:
    claim_id: str
    claim_text: str
    status: CitationVerificationStatus
    citation_ids: list[str]
    supporting_chunk_ids: list[str]
    unsupported_reason: Optional[str]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass
class CitationVerificationResult:
    status: CitationVerificationStatus
    verified: bool
    claim_verifications: list[ClaimVerification] = field(default_factory=list)
    valid_citations: list[CitationRecord] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    unsupported_claim_count: int = 0
    supported_claim_count: int = 0
    citation_coverage: float = 0.0
    unsupported_claim_ratio: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "verified": self.verified,
            "claim_verifications": [item.to_dict() for item in self.claim_verifications],
            "valid_citations": [item.to_dict() for item in self.valid_citations],
            "invalid_citations": list(self.invalid_citations),
            "unsupported_claim_count": self.unsupported_claim_count,
            "supported_claim_count": self.supported_claim_count,
            "citation_coverage": round(self.citation_coverage, 4),
            "unsupported_claim_ratio": round(self.unsupported_claim_ratio, 4),
            "warnings": list(self.warnings),
        }


class CitationVerifier:
    MARKER_RE = re.compile(r"\[C(\d+)\]", re.IGNORECASE)
    TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]")
    NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:%|％)?")

    def __init__(self, min_support_score: float = 0.35, acl_filter: Optional[ACLFilter] = None) -> None:
        self.min_support_score = max(0.0, min(1.0, float(min_support_score)))
        self.acl_filter = acl_filter or ACLFilter()

    def verify(
        self,
        *,
        answer: str,
        claims: Iterable[Claim],
        contexts: Iterable[Any],
        citations: Optional[Iterable[Any]] = None,
        request_context: Optional[RequestContext] = None,
    ) -> CitationVerificationResult:
        context_list = list(contexts)
        context_records = [CitationRecord.from_context(context) for context in context_list]
        context_by_id = {record.citation_id: record for record in context_records}
        context_text_by_id = {
            record.citation_id: str(getattr(context, "text", "") or record.quoted_span)
            for record, context in zip(context_records, context_list)
        }
        context_chunk_ids = {record.chunk_id for record in context_records}
        supplied = list(citations) if citations is not None else context_records
        supplied_records = [self._to_record(item, index) for index, item in enumerate(supplied, start=1)]

        valid_citations: list[CitationRecord] = []
        invalid_citations: list[str] = []
        warnings: list[str] = []
        for supplied_record in supplied_records:
            canonical = context_by_id.get(supplied_record.citation_id)
            if canonical is None or supplied_record.chunk_id not in context_chunk_ids:
                invalid_citations.append(supplied_record.citation_id)
                continue
            if not self._page_is_valid(canonical):
                invalid_citations.append(canonical.citation_id)
                continue
            if request_context is not None and not self.acl_filter.policy.can_access(canonical.metadata, request_context):
                warnings.append("UNAUTHORIZED_CITATION_REMOVED")
                continue
            valid_citations.append(canonical)

        answer_marker_ids = {f"C{number}" for number in self.MARKER_RE.findall(answer or "")}
        known_ids = {record.citation_id for record in valid_citations}
        invalid_citations.extend(sorted(answer_marker_ids - known_ids))
        invalid_citations = _dedupe(invalid_citations)
        if invalid_citations:
            warnings.append("INVALID_CITATION")
        claims_list = list(claims)
        claim_verifications: list[ClaimVerification] = []
        cited_claims = 0
        supported_count = 0
        unsupported_count = 0

        for claim in claims_list:
            claim_citations = list(claim.citation_ids)
            if claim_citations:
                cited_claims += 1
            if not claim_citations:
                unsupported_count += 1
                claim_verifications.append(
                    ClaimVerification(
                        claim_id=claim.claim_id,
                        claim_text=claim.text,
                        status=CitationVerificationStatus.UNSUPPORTED,
                        citation_ids=[],
                        supporting_chunk_ids=[],
                        unsupported_reason="MISSING_CITATION_MARKER",
                        confidence=0.0,
                    )
                )
                continue

            candidate_records = [context_by_id[cid] for cid in claim_citations if cid in context_by_id and cid in known_ids]
            if not candidate_records or any(cid not in known_ids for cid in claim_citations):
                unsupported_count += 1
                claim_verifications.append(
                    ClaimVerification(
                        claim_id=claim.claim_id,
                        claim_text=claim.text,
                        status=CitationVerificationStatus.INVALID_CITATION,
                        citation_ids=claim_citations,
                        supporting_chunk_ids=[],
                        unsupported_reason="INVALID_CITATION",
                        confidence=0.0,
                    )
                )
                continue

            scored = [
                (
                    self._support_score(claim.text, context_text_by_id.get(record.citation_id, record.quoted_span)),
                    record,
                )
                for record in candidate_records
            ]
            best_score, best_record = max(scored, key=lambda item: item[0])
            best_evidence = context_text_by_id.get(best_record.citation_id, best_record.quoted_span)
            status = (
                CitationVerificationStatus.VERIFIED
                if best_score >= self.min_support_score and self._numbers_supported(claim.text, best_evidence)
                else CitationVerificationStatus.PARTIALLY_SUPPORTED
                if best_score > 0
                else CitationVerificationStatus.UNSUPPORTED
            )
            if status == CitationVerificationStatus.UNSUPPORTED:
                unsupported_count += 1
                reason = "NO_SUPPORTING_CHUNK"
            else:
                supported_count += 1
                reason = None if status == CitationVerificationStatus.VERIFIED else "PARTIAL_TEXT_OR_NUMERIC_OVERLAP"
            claim_verifications.append(
                ClaimVerification(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    status=status,
                    citation_ids=claim_citations,
                    supporting_chunk_ids=[best_record.chunk_id] if best_score > 0 else [],
                    unsupported_reason=reason,
                    confidence=round(best_score, 4),
                )
            )

        if claims_list and cited_claims == 0 and context_records:
            warnings.append("MISSING_CITATION_MARKER")
        if unsupported_count:
            warnings.append("UNSUPPORTED_CLAIM")
        coverage = cited_claims / len(claims_list) if claims_list else (1.0 if answer_marker_ids else 0.0 if answer else 1.0)
        unsupported_ratio = unsupported_count / len(claims_list) if claims_list else 0.0
        if invalid_citations:
            status = CitationVerificationStatus.INVALID_CITATION
        elif unsupported_count:
            status = CitationVerificationStatus.UNSUPPORTED
        elif any(item.status == CitationVerificationStatus.PARTIALLY_SUPPORTED for item in claim_verifications):
            status = CitationVerificationStatus.PARTIALLY_SUPPORTED
        else:
            status = CitationVerificationStatus.VERIFIED
        return CitationVerificationResult(
            status=status,
            verified=status == CitationVerificationStatus.VERIFIED,
            claim_verifications=claim_verifications,
            valid_citations=_dedupe_records(valid_citations),
            invalid_citations=invalid_citations,
            unsupported_claim_count=unsupported_count,
            supported_claim_count=supported_count,
            citation_coverage=max(0.0, min(1.0, coverage)),
            unsupported_claim_ratio=max(0.0, min(1.0, unsupported_ratio)),
            warnings=_dedupe(warnings),
        )

    def _to_record(self, item: Any, index: int) -> CitationRecord:
        if isinstance(item, CitationRecord):
            return item
        if isinstance(item, Mapping):
            return CitationRecord(
                citation_id=str(item.get("citation_id") or f"C{index}"),
                document_id=item.get("document_id"),
                version_id=item.get("version_id"),
                chunk_id=str(item.get("chunk_id") or ""),
                source_uri=item.get("source_uri") or item.get("source"),
                source_title=item.get("source_title") or item.get("title"),
                page_start=_int(item.get("page_start", item.get("page"))),
                page_end=_int(item.get("page_end", item.get("page"))),
                quoted_span=str(item.get("quoted_span") or item.get("snippet") or item.get("text") or ""),
                confidence=float(item.get("confidence", item.get("score", 0.0)) or 0.0),
                metadata=dict(item.get("metadata") or {}),
            )
        return CitationRecord.from_context(item)

    @staticmethod
    def _page_is_valid(record: CitationRecord) -> bool:
        return all(page is None or page > 0 for page in (record.page_start, record.page_end))

    def _support_score(self, claim: str, evidence: str) -> float:
        claim_clean = self.MARKER_RE.sub("", claim).strip().lower()
        evidence_clean = (evidence or "").lower()
        if claim_clean and claim_clean in evidence_clean:
            return 1.0
        claim_tokens = set(self.TOKEN_RE.findall(claim_clean))
        evidence_tokens = set(self.TOKEN_RE.findall(evidence_clean))
        claim_tokens -= {"的", "是", "为", "了", "在", "和", "与", "the", "is", "a", "an", "of", "to"}
        overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
        number_overlap = self._number_overlap(claim, evidence)
        return max(0.0, min(1.0, overlap * 0.7 + number_overlap * 0.3))

    def _numbers_supported(self, claim: str, evidence: str) -> bool:
        claim_numbers = set(self.NUMBER_RE.findall(claim))
        return not claim_numbers or claim_numbers.issubset(set(self.NUMBER_RE.findall(evidence)))

    def _number_overlap(self, claim: str, evidence: str) -> float:
        claim_numbers = set(self.NUMBER_RE.findall(claim))
        if not claim_numbers:
            return 1.0
        evidence_numbers = set(self.NUMBER_RE.findall(evidence))
        return len(claim_numbers & evidence_numbers) / len(claim_numbers)


def _int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _dedupe_records(items: list[CitationRecord]) -> list[CitationRecord]:
    seen: set[str] = set()
    output: list[CitationRecord] = []
    for item in items:
        if item.citation_id not in seen:
            seen.add(item.citation_id)
            output.append(item)
    return output
