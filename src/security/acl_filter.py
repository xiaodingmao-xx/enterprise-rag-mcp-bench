"""Recall-stage and post-filter ACL integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.security.context import RequestContext
from src.security.policy import ACLPolicy


@dataclass(frozen=True)
class ACLFilterResult:
    results: list[Any]
    input_count: int
    filtered_count: int
    post_filter: bool = True
    potential_recall_loss: bool = False


class ACLFilter:
    """Build backend-safe tenant filters and enforce complete ACL post-filters."""

    def __init__(self, policy: ACLPolicy | None = None) -> None:
        self.policy = policy or ACLPolicy()

    @staticmethod
    def native_filters(context: RequestContext) -> Dict[str, Any]:
        """Tenant equality is supported by Chroma and most metadata backends."""

        return {"tenant_id": context.tenant_id} if context.tenant_id else {}

    def filter_results(self, results: list[Any], context: RequestContext, *, overfetch_limit: int | None = None) -> ACLFilterResult:
        input_count = len(results)
        filtered = self.policy.filter_retrieval_results(results, context)
        potential_loss = bool(overfetch_limit is not None and input_count >= overfetch_limit and len(filtered) < input_count)
        return ACLFilterResult(
            results=filtered,
            input_count=input_count,
            filtered_count=len(filtered),
            post_filter=True,
            potential_recall_loss=potential_loss,
        )

    def filter_records(self, records: list[dict[str, Any]], context: RequestContext) -> ACLFilterResult:
        filtered = self.policy.filter_records(records, context)
        return ACLFilterResult(
            results=filtered,
            input_count=len(records),
            filtered_count=len(filtered),
            post_filter=True,
            potential_recall_loss=False,
        )

