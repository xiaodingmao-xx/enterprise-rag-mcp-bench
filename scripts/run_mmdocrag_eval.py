#!/usr/bin/env python
"""Run compact MMDocRAG evaluation across retrieval modes.

The script keeps the existing RAG retrieval chain intact and adds multimodal
document QA metrics on top of retrieved chunks:

- recall@k and ndcg@k
- modality_recall@k, image_hit@k, table_hit@k
- answer_correctness and faithfulness when LLM judge is enabled
- citation_accuracy
- latency metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ablation_eval import (  # noqa: E402
    MODES,
    _build_components,
    _chunk_ids,
    _run_retrieval,
    _serialise_result,
    _source_labels,
)
from src.core.settings import load_settings  # noqa: E402
from src.observability.evaluation.citation_metrics import citation_accuracy  # noqa: E402
from src.observability.evaluation.generation_metrics import (  # noqa: E402
    JudgeMetricResult,
    answer_correctness,
    faithfulness,
)
from src.observability.evaluation.ir_metrics import evaluate_ranking_at_k  # noqa: E402
from src.observability.evaluation.multimodal_metrics import (  # noqa: E402
    evaluate_multimodal_at_k,
)


@dataclass(frozen=True)
class MMDocRAGTestCase:
    """Optional-field golden test case for multimodal document RAG."""

    query: str
    question_type: str | None = None
    reference_answer: str | None = None
    generated_answer: str | None = None
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    expected_pages: list[Any] = field(default_factory=list)
    expected_modalities: list[str] = field(default_factory=list)
    expected_evidence: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MMDocRAGTestCase":
        """Create a test case while preserving backward compatibility."""

        return cls(
            query=str(data["query"]),
            question_type=data.get("question_type"),
            reference_answer=data.get("reference_answer"),
            generated_answer=data.get("generated_answer"),
            expected_chunk_ids=[str(item) for item in data.get("expected_chunk_ids", [])],
            expected_sources=[str(item) for item in data.get("expected_sources", [])],
            expected_pages=list(data.get("expected_pages", [])),
            expected_modalities=[
                str(item) for item in data.get("expected_modalities", [])
            ],
            expected_evidence=[
                item for item in data.get("expected_evidence", []) if isinstance(item, dict)
            ],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMDocRAG evaluation with retrieval, multimodal, generation, citation, and latency metrics.",
    )
    parser.add_argument(
        "--dataset",
        default="tests/fixtures/mmdocrag_golden_test_set.json",
        help="MMDocRAG golden test set JSON path.",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Settings YAML path.",
    )
    parser.add_argument("--collection", default="default", help="Collection name.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for metrics.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=0,
        help="Candidate count before rerank. Defaults to 2 * top-k.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="Retrieval modes to evaluate.",
    )
    parser.add_argument(
        "--enable-generation",
        action="store_true",
        help="Generate a lightweight extractive answer from retrieved contexts.",
    )
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="Enable LLM-as-Judge metrics when an LLM provider is configured.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results",
        help="Directory for JSON result files.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print and save a Markdown summary table next to the JSON file.",
    )
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def load_mmdocrag_test_set(path: str | Path) -> list[MMDocRAGTestCase]:
    """Load a backward-compatible MMDocRAG test set."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw_cases = data.get("test_cases", data if isinstance(data, list) else None)
    if not isinstance(raw_cases, list):
        raise ValueError("Invalid MMDocRAG test set: expected list or 'test_cases'.")

    return [MMDocRAGTestCase.from_dict(item) for item in raw_cases]


def _normalise_path(value: str) -> str:
    return str(value).strip().replace("\\", "/").lower()


def _basename(value: str) -> str:
    return PurePosixPath(_normalise_path(value)).name


def _expected_sources_use_basename(expected_sources: Iterable[str]) -> bool:
    normalised = [_normalise_path(item) for item in expected_sources if str(item).strip()]
    return bool(normalised) and all("/" not in item for item in normalised)


def _evidence_chunk_ids(test_case: MMDocRAGTestCase) -> list[str]:
    return [
        str(item["chunk_id"])
        for item in test_case.expected_evidence
        if item.get("chunk_id")
    ]


def _evidence_sources(test_case: MMDocRAGTestCase) -> list[str]:
    return [
        str(item["source"])
        for item in test_case.expected_evidence
        if item.get("source")
    ]


def _relevance_labels(test_case: MMDocRAGTestCase) -> tuple[str, list[str]]:
    if test_case.expected_chunk_ids:
        return "chunk_id", [str(item) for item in test_case.expected_chunk_ids]

    evidence_chunk_ids = _evidence_chunk_ids(test_case)
    if evidence_chunk_ids:
        return "chunk_id", evidence_chunk_ids

    expected_sources = test_case.expected_sources or _evidence_sources(test_case)
    if expected_sources:
        if _expected_sources_use_basename(expected_sources):
            return "source", [_basename(item) for item in expected_sources]
        return "source", [_normalise_path(item) for item in expected_sources]

    return "none", []


def _result_texts(results: Sequence[Any]) -> list[str]:
    texts: list[str] = []
    for result in results:
        if isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = getattr(result, "text", "")
        if str(text).strip():
            texts.append(str(text))
    return texts


def _generate_extractive_answer(results: Sequence[Any], max_contexts: int = 3) -> str:
    parts: list[str] = []
    for index, text in enumerate(_result_texts(results)[:max_contexts], start=1):
        cleaned = " ".join(text.split())
        if cleaned:
            parts.append(f"[{index}] {cleaned[:500]}")
    return "\n".join(parts)


def _aggregate_query_metrics(query_results: Sequence[dict[str, Any]]) -> dict[str, float]:
    metric_values: dict[str, list[float]] = {}
    for item in query_results:
        for key, value in item.get("metrics", {}).items():
            if value is None:
                continue
            metric_values.setdefault(key, []).append(float(value))

    return {
        key: sum(values) / len(values)
        for key, values in sorted(metric_values.items())
        if values
    }


def _metric_result_to_query_fields(
    metric_name: str,
    result: JudgeMetricResult,
    metrics: dict[str, float],
    metric_reasons: dict[str, str],
    skipped_metrics: list[str],
) -> None:
    if result.score is None:
        skipped_metrics.append(metric_name)
        metric_reasons[metric_name] = result.error or result.reason
        return

    metrics[metric_name] = result.score
    if result.reason:
        metric_reasons[metric_name] = result.reason


def _create_judge(settings: Any, enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        import src.libs.llm  # noqa: F401
        from src.libs.llm.llm_factory import LLMFactory

        return LLMFactory.create(settings)
    except Exception as exc:
        print(f"[WARN] LLM judge disabled: {exc}", file=sys.stderr)
        return None


def _evaluate_mode(
    *,
    mode: str,
    test_cases: Sequence[MMDocRAGTestCase],
    top_k: int,
    candidate_k: int,
    components: dict[str, Any],
    enable_generation: bool,
    judge: Any | None,
) -> dict[str, Any]:
    mode_t0 = time.monotonic()
    query_results: list[dict[str, Any]] = []

    for index, test_case in enumerate(test_cases, start=1):
        query_t0 = time.monotonic()
        retrieval_t0 = time.monotonic()
        retrieved_results: list[Any] = []
        error: str | None = None
        try:
            retrieved_results = _run_retrieval(
                mode=mode,
                query=test_case.query,
                top_k=top_k,
                candidate_k=candidate_k,
                components=components,
            )
        except Exception as exc:
            error = str(exc)
        retrieval_latency_ms = (time.monotonic() - retrieval_t0) * 1000.0

        retrieved_ids = _chunk_ids(retrieved_results)
        relevance_type, relevant_labels = _relevance_labels(test_case)
        effective_expected_sources = test_case.expected_sources or _evidence_sources(test_case)
        retrieved_sources = _source_labels(retrieved_results, effective_expected_sources)
        retrieved_labels = retrieved_ids if relevance_type == "chunk_id" else retrieved_sources

        metrics: dict[str, float] = {}
        metric_reasons: dict[str, str] = {}
        skipped_metrics: list[str] = []

        if relevant_labels:
            ir_metrics = evaluate_ranking_at_k(retrieved_labels, relevant_labels, top_k)
            metrics[f"recall@{top_k}"] = ir_metrics[f"recall@{top_k}"]
            metrics[f"ndcg@{top_k}"] = ir_metrics[f"ndcg@{top_k}"]
        else:
            skipped_metrics.extend([f"recall@{top_k}", f"ndcg@{top_k}"])

        multimodal_metrics = evaluate_multimodal_at_k(
            retrieved_results,
            test_case.expected_modalities,
            top_k,
        )
        metrics.update(multimodal_metrics)
        if not test_case.expected_modalities:
            skipped_metrics.extend(
                [f"modality_recall@{top_k}", f"image_hit@{top_k}", f"table_hit@{top_k}"]
            )

        generation_t0 = time.monotonic()
        generated_answer = test_case.generated_answer
        if generated_answer is None and enable_generation:
            generated_answer = _generate_extractive_answer(retrieved_results)
        generation_latency_ms = (time.monotonic() - generation_t0) * 1000.0

        contexts = _result_texts(retrieved_results)
        if judge is not None and generated_answer is not None:
            correctness = answer_correctness(
                query=test_case.query,
                generated_answer=generated_answer,
                reference_answer=test_case.reference_answer,
                retrieved_contexts=contexts,
                judge=judge,
            )
            _metric_result_to_query_fields(
                "answer_correctness",
                correctness,
                metrics,
                metric_reasons,
                skipped_metrics,
            )

            faithful = faithfulness(
                query=test_case.query,
                generated_answer=generated_answer,
                retrieved_contexts=contexts,
                judge=judge,
            )
            _metric_result_to_query_fields(
                "faithfulness",
                faithful,
                metrics,
                metric_reasons,
                skipped_metrics,
            )
        else:
            skipped_metrics.extend(["answer_correctness", "faithfulness"])

        citation_score = citation_accuracy(
            generated_answer=generated_answer,
            retrieved_results=retrieved_results,
            expected_sources=test_case.expected_sources,
            expected_pages=test_case.expected_pages,
            expected_chunk_ids=test_case.expected_chunk_ids,
            expected_evidence=test_case.expected_evidence,
        )
        if citation_score is None:
            skipped_metrics.append("citation_accuracy")
        else:
            metrics["citation_accuracy"] = citation_score

        end_to_end_latency_ms = (time.monotonic() - query_t0) * 1000.0
        metrics["retrieval_latency_ms"] = retrieval_latency_ms
        metrics["generation_latency_ms"] = generation_latency_ms
        metrics["end_to_end_latency_ms"] = end_to_end_latency_ms

        query_results.append(
            {
                "index": index,
                "query": test_case.query,
                "question_type": test_case.question_type,
                "relevance_type": relevance_type,
                "relevant_labels": relevant_labels,
                "expected_modalities": test_case.expected_modalities,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_sources": retrieved_sources,
                "generated_answer": generated_answer,
                "metrics": {key: round(value, 4) for key, value in metrics.items()},
                "metric_reasons": metric_reasons,
                "skipped_metrics": sorted(set(skipped_metrics)),
                "error": error,
                "results": [_serialise_result(result) for result in retrieved_results],
            }
        )

    aggregate_metrics = _aggregate_query_metrics(query_results)
    evaluated_query_count = sum(
        1
        for item in query_results
        if item["metrics"].get(f"recall@{top_k}") is not None
        or item["metrics"].get(f"ndcg@{top_k}") is not None
    )

    return {
        "mode": mode,
        "query_count": len(query_results),
        "evaluated_query_count": evaluated_query_count,
        "aggregate_metrics": {
            key: round(value, 4) for key, value in aggregate_metrics.items()
        },
        "elapsed_ms": round((time.monotonic() - mode_t0) * 1000.0, 1),
        "query_results": query_results,
    }


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def format_markdown_table(report: dict[str, Any], top_k: int) -> str:
    """Format aggregate MMDocRAG metrics as a Markdown table."""

    headers = [
        "Mode",
        "Evaluated",
        f"Recall@{top_k}",
        f"NDCG@{top_k}",
        f"Modality Recall@{top_k}",
        f"Image Hit@{top_k}",
        f"Table Hit@{top_k}",
        "Answer Correctness",
        "Faithfulness",
        "Citation Accuracy",
        "Latency ms",
    ]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for mode in report["modes"]:
        mode_result = report["results"][mode]
        aggregate = mode_result.get("aggregate_metrics", {})
        rows.append(
            "| "
            + " | ".join(
                [
                    mode,
                    str(mode_result.get("evaluated_query_count", 0)),
                    _format_metric(aggregate.get(f"recall@{top_k}")),
                    _format_metric(aggregate.get(f"ndcg@{top_k}")),
                    _format_metric(aggregate.get(f"modality_recall@{top_k}")),
                    _format_metric(aggregate.get(f"image_hit@{top_k}")),
                    _format_metric(aggregate.get(f"table_hit@{top_k}")),
                    _format_metric(aggregate.get("answer_correctness")),
                    _format_metric(aggregate.get("faithfulness")),
                    _format_metric(aggregate.get("citation_accuracy")),
                    _format_metric(aggregate.get("end_to_end_latency_ms"), digits=1),
                ]
            )
            + " |"
        )

    return "\n".join(rows)


def run_mmdocrag_eval(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path | None]:
    """Run MMDocRAG evaluation and persist JSON/optional Markdown outputs."""

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")
    if args.candidate_k < 0:
        raise ValueError("--candidate-k must be greater than or equal to 0")

    config_path = _resolve_path(args.config)
    dataset_path = _resolve_path(args.dataset)
    output_dir = _resolve_path(args.output_dir)

    settings = load_settings(config_path)
    test_cases = load_mmdocrag_test_set(dataset_path)
    components = _build_components(settings, args.collection)
    judge = _create_judge(settings, args.enable_llm_judge)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{timestamp}_mmdocrag.json"

    report: dict[str, Any] = {
        "timestamp": timestamp,
        "dataset": str(dataset_path),
        "config": str(config_path),
        "collection": args.collection,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k if args.candidate_k > 0 else args.top_k * 2,
        "modes": list(args.modes),
        "enable_generation": bool(args.enable_generation),
        "enable_llm_judge": bool(args.enable_llm_judge and judge is not None),
        "test_case_count": len(test_cases),
        "results": {},
    }

    total_t0 = time.monotonic()
    for mode in args.modes:
        report["results"][mode] = _evaluate_mode(
            mode=mode,
            test_cases=test_cases,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            components=components,
            enable_generation=args.enable_generation,
            judge=judge,
        )
    report["total_elapsed_ms"] = round((time.monotonic() - total_t0) * 1000.0, 1)

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown_path = None
    if args.markdown:
        markdown = format_markdown_table(report, args.top_k)
        markdown_path = json_path.with_suffix(".md")
        markdown_path.write_text(markdown + "\n", encoding="utf-8")
        print(markdown)

    return report, json_path, markdown_path


def main() -> int:
    args = parse_args()
    try:
        report, json_path, markdown_path = run_mmdocrag_eval(args)
    except Exception as exc:
        print(f"[FAIL] MMDocRAG evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] MMDocRAG results saved: {json_path}")
    if markdown_path is not None:
        print(f"[OK] Markdown table saved: {markdown_path}")
    print(f"[OK] Modes: {', '.join(report['modes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
