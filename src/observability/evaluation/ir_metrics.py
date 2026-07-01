"""Information retrieval metrics for ranking evaluation."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Set


def _validate_k(k: int) -> int:
    if not isinstance(k, int):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    return k


def _normalise_relevant(relevant_ids: Iterable[str]) -> Set[str]:
    return {str(item) for item in relevant_ids if str(item)}


def _top_k_unique(retrieved_ids: Sequence[str], k: int) -> List[str]:
    """Return unique IDs from the first k ranked positions."""

    if k <= 0:
        return []

    seen: set[str] = set()
    unique: list[str] = []
    for item in retrieved_ids[:k]:
        item_id = str(item)
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        unique.append(item_id)
    return unique


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Compute Recall@K for a ranked list."""

    _validate_k(k)
    relevant = _normalise_relevant(relevant_ids)
    if not relevant or k <= 0:
        return 0.0

    hits = sum(1 for item in _top_k_unique(retrieved_ids, k) if item in relevant)
    return hits / len(relevant)


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Compute Precision@K for a ranked list."""

    _validate_k(k)
    relevant = _normalise_relevant(relevant_ids)
    if not relevant or k <= 0:
        return 0.0

    hits = sum(1 for item in _top_k_unique(retrieved_ids, k) if item in relevant)
    return hits / k


def mrr_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Compute reciprocal rank of the first relevant item in top-k."""

    _validate_k(k)
    relevant = _normalise_relevant(relevant_ids)
    if not relevant or k <= 0:
        return 0.0

    for rank, item in enumerate(_top_k_unique(retrieved_ids, k), start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Compute binary NDCG@K for a ranked list."""

    _validate_k(k)
    relevant = _normalise_relevant(relevant_ids)
    if not relevant or k <= 0:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(_top_k_unique(retrieved_ids, k), start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    if ideal_hits <= 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_ranking_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> dict[str, float]:
    """Compute the standard IR metric bundle for one ranked list."""

    return {
        f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        f"mrr@{k}": mrr_at_k(retrieved_ids, relevant_ids, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, relevant_ids, k),
    }
