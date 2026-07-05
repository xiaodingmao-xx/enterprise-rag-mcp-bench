#!/usr/bin/env python
# ruff: noqa: E402,I001
"""Run document-level retrieval evaluation on EnterpriseRAG-Bench.

The script reuses the existing ablation retrieval components and adapts
EnterpriseRAG-Bench ground-truth document IDs (``dsid_xxx``) to retrieved chunk
metadata. It evaluates dense, BM25, hybrid, and hybrid_rerank retrieval modes.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.core.settings import load_settings  # noqa: E402
from src.observability.evaluation.ir_metrics import evaluate_ranking_at_k  # noqa: E402


DSID_PATTERN = re.compile(r"(dsid_[0-9a-fA-F]+)", re.IGNORECASE)
SOURCE_METADATA_KEYS = (
    "source_path",
    "source",
    "file_path",
    "source_file",
    "path",
)
MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@lru_cache(maxsize=1)
def _ablation_helpers() -> tuple[Any, Any, Any, Any]:
    from scripts.run_ablation_eval import (
        MODES,
        _build_components,
        _run_retrieval,
        _serialise_result,
    )

    return MODES, _build_components, _run_retrieval, _serialise_result


def _available_modes() -> tuple[str, ...]:
    return MODES


@dataclass(frozen=True)
class EnterpriseRAGTestCase:
    """Single EnterpriseRAG-Bench retrieval test case."""

    question_id: str
    query: str
    question_type: str | None
    source_types: list[str]
    expected_doc_ids: list[str]
    gold_answer: str | None = None
    answer_facts: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run document-level retrieval evaluation on EnterpriseRAG-Bench "
            "for dense, BM25, hybrid, and hybrid_rerank modes."
        ),
    )
    parser.add_argument(
        "--questions-file",
        "--dataset",
        dest="questions_file",
        required=True,
        help=(
            "EnterpriseRAG-Bench questions JSONL path. A dataset root directory "
            "or data/questions/test.parquet is also accepted."
        ),
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Settings YAML path. Defaults to config/settings.yaml.",
    )
    parser.add_argument(
        "--collection",
        default="enterprise_rag",
        help="Collection name containing ingested EnterpriseRAG-Bench documents.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K for document-level IR metrics.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=0,
        help="Candidate count before hybrid_rerank. Defaults to 2 * top-k.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=_available_modes(),
        default=list(_available_modes()),
        help="Retrieval modes to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results",
        help="Directory for JSON and optional Markdown reports.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print and save a Markdown summary table next to the JSON report.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Evaluate only the first N questions after filtering.",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=None,
        help="Only evaluate these EnterpriseRAG-Bench question_type values.",
    )
    parser.add_argument(
        "--source-types",
        nargs="+",
        default=None,
        help="Only evaluate questions whose source_types intersect these values.",
    )
    parser.add_argument(
        "--skip-no-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip questions without expected_doc_ids. Enabled by default.",
    )
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def _normalise_filter_values(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalised = {str(value).strip().lower() for value in values if str(value).strip()}
    return normalised or None


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []

    if hasattr(value, "tolist"):
        return _coerce_str_list(value.tolist())

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                except Exception:
                    continue
                return _coerce_str_list(parsed)
        return [stripped]

    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):  # noqa: UP038
        return [str(item).strip() for item in value if str(item).strip()]

    return [str(value).strip()] if str(value).strip() else []


def extract_dsid(value: str) -> str | None:
    """Extract an EnterpriseRAG-Bench ``dsid_xxx`` document ID from text."""

    match = DSID_PATTERN.search(str(value or ""))
    if match is None:
        return None
    return match.group(1).lower()


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    """Deduplicate non-empty strings while preserving first-seen order."""

    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _normalise_expected_doc_ids(value: Any) -> list[str]:
    doc_ids: list[str] = []
    for item in _coerce_str_list(value):
        dsid = extract_dsid(item)
        doc_ids.append(dsid or item.strip().lower())
    return dedupe_keep_order(doc_ids)


def _question_records_from_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: expected object")
            yield record


def _question_records_from_json(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw_records = data.get("questions", data) if isinstance(data, dict) else data
    if not isinstance(raw_records, list):
        raise ValueError("Invalid questions JSON: expected a list or {'questions': [...]}")

    for index, record in enumerate(raw_records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid questions JSON at index {index}: expected object")
        yield record


def _question_records_from_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet requires the optional 'datasets' package. "
            "Export EnterpriseRAG-Bench questions to JSONL or install project dependencies."
        ) from exc

    try:
        dataset = load_dataset("parquet", data_files=str(path), split="train")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read EnterpriseRAG-Bench parquet questions from {path}: {exc}"
        ) from exc

    for record in dataset:
        yield dict(record)


def _resolve_questions_file(path: str | Path) -> Path:
    path_obj = _resolve_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"EnterpriseRAG-Bench questions file not found: {path_obj}"
        )

    if path_obj.is_dir():
        candidates = [
            path_obj / "questions.jsonl",
            path_obj / "data" / "questions" / "test.parquet",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not find questions.jsonl or data/questions/test.parquet under "
            f"EnterpriseRAG-Bench directory: {path_obj}"
        )

    return path_obj


def _iter_question_records(path: str | Path) -> Iterator[dict[str, Any]]:
    questions_path = _resolve_questions_file(path)
    suffix = questions_path.suffix.lower()
    if suffix == ".jsonl":
        yield from _question_records_from_jsonl(questions_path)
        return
    if suffix == ".json":
        yield from _question_records_from_json(questions_path)
        return
    if suffix == ".parquet":
        yield from _question_records_from_parquet(questions_path)
        return
    raise ValueError(
        f"Unsupported questions file type: {questions_path.suffix}. "
        "Use .jsonl, .json, .parquet, or pass the EnterpriseRAG-Bench root directory."
    )


def load_enterprise_questions(
    path: str | Path,
    *,
    max_questions: int | None = None,
    question_types: set[str] | None = None,
    source_types: set[str] | None = None,
    skip_no_ground_truth: bool = True,
) -> list[EnterpriseRAGTestCase]:
    """Load EnterpriseRAG-Bench questions with lightweight filtering."""

    question_type_filter = _normalise_filter_values(question_types)
    source_type_filter = _normalise_filter_values(source_types)
    test_cases: list[EnterpriseRAGTestCase] = []

    for row_index, record in enumerate(_iter_question_records(path), start=1):
        question_type = record.get("question_type")
        question_type_text = (
            str(question_type).strip() if question_type is not None else None
        )
        if question_type_filter and (
            question_type_text is None
            or question_type_text.lower() not in question_type_filter
        ):
            continue

        current_source_types = _coerce_str_list(record.get("source_types"))
        if source_type_filter:
            current_source_type_set = {
                source_type.lower() for source_type in current_source_types
            }
            if not current_source_type_set.intersection(source_type_filter):
                continue

        expected_doc_ids = _normalise_expected_doc_ids(record.get("expected_doc_ids"))
        if skip_no_ground_truth and not expected_doc_ids:
            continue

        query = str(record.get("question") or record.get("query") or "").strip()
        if not query:
            continue

        question_id = str(record.get("question_id") or record.get("id") or row_index)
        test_cases.append(
            EnterpriseRAGTestCase(
                question_id=question_id,
                query=query,
                question_type=question_type_text,
                source_types=current_source_types,
                expected_doc_ids=expected_doc_ids,
                gold_answer=(
                    str(record["gold_answer"])
                    if record.get("gold_answer") is not None
                    else None
                ),
                answer_facts=_coerce_str_list(record.get("answer_facts")),
            )
        )

        if max_questions is not None and len(test_cases) >= max_questions:
            break

    return test_cases


def _result_metadata(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        metadata = result.get("metadata", {})
    else:
        metadata = getattr(result, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _result_field(result: Any, key: str) -> Any:
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def _extract_dsid_from_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for nested_value in value.values():
            dsid = _extract_dsid_from_value(nested_value)
            if dsid:
                return dsid
        return None
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):  # noqa: UP038
        for nested_value in value:
            dsid = _extract_dsid_from_value(nested_value)
            if dsid:
                return dsid
        return None
    return extract_dsid(str(value))


def result_to_doc_id(result: Any) -> str:
    """Map one retrieval result to an EnterpriseRAG-Bench document ID."""

    metadata = _result_metadata(result)
    for key in SOURCE_METADATA_KEYS:
        dsid = _extract_dsid_from_value(metadata.get(key))
        if dsid:
            return dsid

    for key in ("chunk_id", "id"):
        dsid = _extract_dsid_from_value(_result_field(result, key))
        if dsid:
            return dsid

    return ""


def _retrieved_doc_ids(results: Sequence[Any]) -> list[str]:
    return dedupe_keep_order(result_to_doc_id(result) for result in results)


def _empty_metrics(top_k: int, latency_ms: float, gold_count: int, retrieved_count: int) -> dict[str, float]:
    return {
        f"recall@{top_k}": 0.0,
        f"precision@{top_k}": 0.0,
        f"mrr@{top_k}": 0.0,
        f"ndcg@{top_k}": 0.0,
        f"hit@{top_k}": 0.0,
        "gold_doc_count": float(gold_count),
        "retrieved_doc_count": float(retrieved_count),
        "latency_ms": round(latency_ms, 1),
    }


def _evaluate_query_metrics(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    top_k: int,
    latency_ms: float,
) -> dict[str, float]:
    metrics = evaluate_ranking_at_k(retrieved_doc_ids, expected_doc_ids, top_k)
    expected_set = set(expected_doc_ids)
    metrics[f"hit@{top_k}"] = (
        1.0 if any(doc_id in expected_set for doc_id in retrieved_doc_ids[:top_k]) else 0.0
    )
    metrics["gold_doc_count"] = float(len(expected_doc_ids))
    metrics["retrieved_doc_count"] = float(len(retrieved_doc_ids))
    metrics["latency_ms"] = round(latency_ms, 1)
    return {key: round(float(value), 4) for key, value in metrics.items()}


def _aggregate_query_metrics(
    query_results: Sequence[dict[str, Any]],
    top_k: int,
) -> dict[str, float]:
    metric_keys = [
        f"recall@{top_k}",
        f"precision@{top_k}",
        f"mrr@{top_k}",
        f"ndcg@{top_k}",
        f"hit@{top_k}",
        "latency_ms",
    ]
    values_by_key: dict[str, list[float]] = {key: [] for key in metric_keys}

    for item in query_results:
        if item.get("skipped"):
            continue
        metrics = item.get("metrics", {})
        for key in metric_keys:
            if key in metrics:
                values_by_key[key].append(float(metrics[key]))

    return {
        key: round(sum(values) / len(values), 4) if values else 0.0
        for key, values in values_by_key.items()
    }


def _evaluate_mode(
    *,
    mode: str,
    test_cases: Sequence[EnterpriseRAGTestCase],
    top_k: int,
    candidate_k: int,
    components: dict[str, Any],
) -> dict[str, Any]:
    mode_t0 = time.monotonic()
    query_results: list[dict[str, Any]] = []

    for index, test_case in enumerate(test_cases, start=1):
        query_t0 = time.monotonic()
        retrieved_results: list[Any] = []
        error: str | None = None

        try:
            _, _, run_retrieval, serialise_result = _ablation_helpers()
            retrieved_results = run_retrieval(
                mode=mode,
                query=test_case.query,
                top_k=top_k,
                candidate_k=candidate_k,
                components=components,
            )
        except Exception as exc:
            error = str(exc)

        latency_ms = (time.monotonic() - query_t0) * 1000.0
        retrieved_doc_ids = _retrieved_doc_ids(retrieved_results)
        skipped = not test_case.expected_doc_ids

        if skipped or error:
            metrics = _empty_metrics(
                top_k,
                latency_ms,
                len(test_case.expected_doc_ids),
                len(retrieved_doc_ids),
            )
        else:
            metrics = _evaluate_query_metrics(
                retrieved_doc_ids,
                test_case.expected_doc_ids,
                top_k,
                latency_ms,
            )

        query_results.append(
            {
                "index": index,
                "question_id": test_case.question_id,
                "question_type": test_case.question_type,
                "source_types": test_case.source_types,
                "query": test_case.query,
                "expected_doc_ids": test_case.expected_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "metrics": metrics,
                "skipped": skipped,
                "error": error,
                "results": [serialise_result(result) for result in retrieved_results],
            }
        )

    evaluated_count = sum(1 for item in query_results if not item.get("skipped"))
    error_count = sum(1 for item in query_results if item.get("error"))

    return {
        "mode": mode,
        "query_count": len(query_results),
        "evaluated_query_count": evaluated_count,
        "skipped_query_count": len(query_results) - evaluated_count,
        "error_count": error_count,
        "aggregate_metrics": _aggregate_query_metrics(query_results, top_k),
        "elapsed_ms": round((time.monotonic() - mode_t0) * 1000.0, 1),
        "query_results": query_results,
    }


def _collection_record_count(components: dict[str, Any]) -> int | None:
    dense_retriever = components.get("dense_retriever")
    vector_store = getattr(dense_retriever, "vector_store", None)
    if vector_store is None:
        return None

    get_collection_stats = getattr(vector_store, "get_collection_stats", None)
    if callable(get_collection_stats):
        try:
            stats = get_collection_stats()
            return int(stats.get("count", 0))
        except Exception:
            return None

    collection = getattr(vector_store, "collection", None)
    count = getattr(collection, "count", None)
    if callable(count):
        try:
            return int(count())
        except Exception:
            return None

    return None


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def format_markdown_table(report: dict[str, Any], top_k: int) -> str:
    """Format aggregate EnterpriseRAG-Bench retrieval metrics as Markdown."""

    headers = [
        "Mode",
        "Evaluated",
        f"Recall@{top_k}",
        f"Precision@{top_k}",
        f"MRR@{top_k}",
        f"NDCG@{top_k}",
        f"Hit@{top_k}",
        "Avg Latency ms",
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
                    _format_metric(aggregate.get(f"precision@{top_k}")),
                    _format_metric(aggregate.get(f"mrr@{top_k}")),
                    _format_metric(aggregate.get(f"ndcg@{top_k}")),
                    _format_metric(aggregate.get(f"hit@{top_k}")),
                    _format_metric(aggregate.get("latency_ms"), digits=1),
                ]
            )
            + " |"
        )

    return "\n".join(rows)


def run_enterprise_rag_eval(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, Path | None]:
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")
    if args.candidate_k < 0:
        raise ValueError("--candidate-k must be greater than or equal to 0")
    if args.max_questions is not None and args.max_questions <= 0:
        raise ValueError("--max-questions must be greater than 0 when provided")

    config_path = _resolve_path(args.config)
    questions_path = _resolve_questions_file(args.questions_file)
    output_dir = _resolve_path(args.output_dir)
    effective_candidate_k = (
        max(args.candidate_k, args.top_k) if args.candidate_k > 0 else args.top_k * 2
    )

    settings = load_settings(config_path)
    test_cases = load_enterprise_questions(
        questions_path,
        max_questions=args.max_questions,
        question_types=set(args.question_types) if args.question_types else None,
        source_types=set(args.source_types) if args.source_types else None,
        skip_no_ground_truth=args.skip_no_ground_truth,
    )
    if not test_cases:
        raise ValueError(
            "No EnterpriseRAG-Bench questions were loaded after filtering. "
            "Check --questions-file, --question-types, --source-types, and "
            "--skip-no-ground-truth."
        )

    try:
        ablation_modes, build_components, _, _ = _ablation_helpers()
        unsupported_modes = [mode for mode in args.modes if mode not in ablation_modes]
        if unsupported_modes:
            raise ValueError(f"Unsupported retrieval modes: {', '.join(unsupported_modes)}")
        components = build_components(settings, args.collection)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize retrieval components for collection "
            f"'{args.collection}'. Ensure documents were ingested first. Detail: {exc}"
        ) from exc

    warnings: list[str] = []
    collection_count = _collection_record_count(components)
    if collection_count == 0:
        warning = (
            f"Collection '{args.collection}' appears to be empty. "
            "Run scripts/ingest.py before evaluating retrieval quality."
        )
        warnings.append(warning)
        print(f"[WARN] {warning}", file=sys.stderr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{timestamp}_enterprise_rag.json"

    report: dict[str, Any] = {
        "timestamp": timestamp,
        "dataset": str(questions_path),
        "config": str(config_path),
        "collection": args.collection,
        "collection_record_count": collection_count,
        "top_k": args.top_k,
        "candidate_k": effective_candidate_k,
        "modes": list(args.modes),
        "test_case_count": len(test_cases),
        "filters": {
            "question_types": list(args.question_types or []),
            "source_types": list(args.source_types or []),
            "max_questions": args.max_questions,
            "skip_no_ground_truth": bool(args.skip_no_ground_truth),
        },
        "warnings": warnings,
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
        report, json_path, markdown_path = run_enterprise_rag_eval(args)
    except Exception as exc:
        print(f"[FAIL] EnterpriseRAG-Bench evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] EnterpriseRAG-Bench results saved: {json_path}")
    if markdown_path is not None:
        print(f"[OK] Markdown table saved: {markdown_path}")
    print(f"[OK] Modes: {', '.join(report['modes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
