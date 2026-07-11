"""Evaluation module for batch quality assessment and IR metrics."""

from src.observability.evaluation.ir_metrics import (
    evaluate_ranking_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.observability.evaluation.multimodal_metrics import (
    evaluate_multimodal_at_k,
    image_hit_at_k,
    modality_recall_at_k,
    table_hit_at_k,
)
from src.observability.evaluation.generation_metrics import (
    JudgeMetricResult,
    answer_correctness,
    faithfulness,
)
from src.observability.evaluation.citation_metrics import citation_accuracy

__all__ = [
    "JudgeMetricResult",
    "answer_correctness",
    "citation_accuracy",
    "evaluate_ranking_at_k",
    "evaluate_multimodal_at_k",
    "faithfulness",
    "image_hit_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "modality_recall_at_k",
    "table_hit_at_k",
]
from src.observability.evaluation.trust_metrics import compute_trust_metrics

__all__ = ["compute_trust_metrics"]
