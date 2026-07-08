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
import hashlib
import json
import re
import sys
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars while evaluating. Enabled by default.",
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
    parser.add_argument(
        "--enable-generation",
        action="store_true",
        help=(
            "Generate grounded answers from retrieved contexts. This is also "
            "enabled automatically when --enable-ragas is used."
        ),
    )
    parser.add_argument(
        "--enable-ragas",
        action="store_true",
        help="Evaluate generated answers with Ragas metrics such as faithfulness.",
    )
    parser.add_argument(
        "--ragas-metrics",
        nargs="+",
        default=["faithfulness"],
        help=(
            "Ragas metrics to compute when --enable-ragas is set. "
            "Defaults to faithfulness."
        ),
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=8000,
        help="Maximum total retrieved context characters used for answer generation.",
    )
    parser.add_argument(
        "--ragas-workers",
        type=int,
        default=1,
        help=(
            "Number of worker threads for answer generation and Ragas evaluation. "
            "Only applies when --enable-generation or --enable-ragas is used."
        ),
    )
    parser.add_argument(
        "--ragas-cache",
        nargs="?",
        const="eval/cache/enterprise_rag_ragas_cache.json",
        default=None,
        help=(
            "Enable a JSON cache for generated answers and Ragas scores. "
            "Optionally pass a cache file path."
        ),
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
    excluded_keys = {"gold_doc_count", "retrieved_doc_count"}
    values_by_key: dict[str, list[float]] = {}

    for item in query_results:
        if item.get("skipped"):
            continue
        metrics = item.get("metrics", {})
        for key, value in metrics.items():
            if key in excluded_keys or value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            values_by_key.setdefault(key, []).append(parsed)

    return {
        key: round(sum(values) / len(values), 4) if values else 0.0
        for key, values in sorted(values_by_key.items())
    }


def _result_texts(results: Sequence[Any], max_chars: int) -> list[str]:
    """Extract non-empty context texts with a total character budget."""

    remaining = max(0, int(max_chars))
    texts: list[str] = []
    if remaining <= 0:
        return texts

    for result in results:
        if isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = getattr(result, "text", "")
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            continue

        if len(cleaned) > remaining:
            cleaned = cleaned[:remaining]
        texts.append(cleaned)
        remaining -= len(cleaned)
        if remaining <= 0:
            break

    return texts


def _create_answer_generator(settings: Any, enabled: bool) -> Any | None:
    if not enabled:
        return None

    try:
        import src.libs.llm  # noqa: F401
        from src.libs.llm.llm_factory import LLMFactory

        return LLMFactory.create(settings)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize answer generation LLM: {exc}") from exc


def _generate_grounded_answer(
    *,
    query: str,
    contexts: Sequence[str],
    answer_generator: Any,
) -> str:
    if answer_generator is None:
        raise RuntimeError("answer generator is not configured")
    if not contexts:
        return "The retrieved contexts do not provide enough information."

    from src.libs.llm.base_llm import Message

    context_block = "\n\n".join(
        f"[{index}] {context}" for index, context in enumerate(contexts, start=1)
    )
    messages = [
        Message(
            role="system",
            content=(
                "You are a strict grounded RAG answer generator. Answer only "
                "from the provided contexts. If the contexts are insufficient, "
                "say that the retrieved contexts do not provide enough "
                "information. Answer in the same language as the question."
            ),
        ),
        Message(
            role="user",
            content=(
                f"Question:\n{query}\n\n"
                f"Retrieved contexts:\n{context_block}\n\n"
                "Grounded answer:"
            ),
        ),
    ]
    response = answer_generator.chat(messages, temperature=0.0)
    return str(getattr(response, "content", response)).strip()


def _create_ragas_evaluator(
    *,
    settings: Any,
    enabled: bool,
    metrics: Sequence[str],
) -> Any | None:
    if not enabled:
        return None

    try:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        return RagasEvaluator(settings=settings, metrics=metrics)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Ragas evaluator: {exc}") from exc


class _NullProgressBar:
    def __enter__(self) -> "_NullProgressBar":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, value: int = 1) -> None:
        return None

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def _progress_bar(*, enabled: bool, total: int, desc: str) -> Any:
    if not enabled:
        return _NullProgressBar()

    try:
        from tqdm import tqdm
    except ImportError:
        return _NullProgressBar()

    return tqdm(
        total=total,
        desc=desc,
        unit="q",
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
    )


class RagasEvalCache:
    """Small JSON cache for expensive generated answers and Ragas scores."""

    VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "version": self.VERSION,
            "entries": {},
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        entries = raw.get("entries")
        if not isinstance(entries, dict):
            return
        self._data = {
            "version": raw.get("version", self.VERSION),
            "entries": entries,
        }

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._data.get("entries", {}).get(key)
            return dict(entry) if isinstance(entry, dict) else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._data.setdefault("entries", {})[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(self.path)


def _ragas_cache_key(
    *,
    mode: str,
    test_case: EnterpriseRAGTestCase,
    retrieved_doc_ids: Sequence[str],
    contexts: Sequence[str],
    ragas_metrics: Sequence[str],
    max_context_chars: int,
    namespace: dict[str, Any] | None = None,
) -> str:
    payload = {
        "version": RagasEvalCache.VERSION,
        "namespace": namespace or {},
        "mode": mode,
        "question_id": test_case.question_id,
        "query": test_case.query,
        "retrieved_doc_ids": list(retrieved_doc_ids),
        "contexts": list(contexts),
        "ragas_metrics": [str(metric).lower() for metric in ragas_metrics],
        "max_context_chars": int(max_context_chars),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_generation_and_ragas(
    *,
    query: str,
    contexts: Sequence[str],
    enable_generation: bool,
    enable_ragas: bool,
    answer_generator: Any | None,
    ragas_evaluator: Any | None,
) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    generated_answer: str | None = None
    generation_error: str | None = None
    generation_latency_ms: float | None = None
    ragas_scores: dict[str, float] = {}
    ragas_error: str | None = None
    ragas_latency_ms: float | None = None

    if enable_generation or enable_ragas:
        generation_t0 = time.monotonic()
        try:
            generated_answer = _generate_grounded_answer(
                query=query,
                contexts=contexts,
                answer_generator=answer_generator,
            )
        except Exception as exc:
            generation_error = str(exc)
        generation_latency_ms = (time.monotonic() - generation_t0) * 1000.0
        metrics["generation_latency_ms"] = round(generation_latency_ms, 1)

    if enable_ragas:
        ragas_t0 = time.monotonic()
        if ragas_evaluator is None:
            ragas_error = "ragas evaluator is not configured"
        elif not generated_answer:
            ragas_error = generation_error or "missing generated answer"
        elif not contexts:
            ragas_error = "missing retrieved contexts"
        else:
            try:
                ragas_scores = ragas_evaluator.evaluate(
                    query=query,
                    retrieved_chunks=[{"text": context} for context in contexts],
                    generated_answer=generated_answer,
                )
                metrics.update(
                    {
                        key: round(float(value), 4)
                        for key, value in ragas_scores.items()
                    }
                )
            except Exception as exc:
                ragas_error = str(exc)
        ragas_latency_ms = (time.monotonic() - ragas_t0) * 1000.0
        metrics["ragas_latency_ms"] = round(ragas_latency_ms, 1)

    return {
        "generated_answer": generated_answer,
        "generation_error": generation_error,
        "generation_latency_ms": generation_latency_ms,
        "ragas": ragas_scores,
        "ragas_error": ragas_error,
        "ragas_latency_ms": ragas_latency_ms,
        "metrics": metrics,
    }


def _evaluate_mode(
    *,
    mode: str,
    test_cases: Sequence[EnterpriseRAGTestCase],
    top_k: int,
    candidate_k: int,
    components: dict[str, Any],
    enable_generation: bool = False,
    enable_ragas: bool = False,
    answer_generator: Any | None = None,
    ragas_evaluator: Any | None = None,
    max_context_chars: int = 8000,
    ragas_workers: int = 1,
    ragas_cache: RagasEvalCache | None = None,
    ragas_metrics: Sequence[str] = (),
    ragas_cache_namespace: dict[str, Any] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    mode_t0 = time.monotonic()
    query_results: list[dict[str, Any]] = []
    postprocess_jobs: list[dict[str, Any]] = []
    _, _, run_retrieval, serialise_result = _ablation_helpers()
    retrieval_error_count = 0
    retrieval_progress = _progress_bar(
        enabled=progress,
        total=len(test_cases),
        desc=f"Retrieval {mode}",
    )

    for index, test_case in enumerate(test_cases, start=1):
        query_t0 = time.monotonic()
        retrieved_results: list[Any] = []
        error: str | None = None

        try:
            retrieved_results = run_retrieval(
                mode=mode,
                query=test_case.query,
                top_k=top_k,
                candidate_k=candidate_k,
                components=components,
            )
        except Exception as exc:
            error = str(exc)
            retrieval_error_count += 1

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

        contexts = _result_texts(retrieved_results, max_context_chars)
        query_results.append(
            {
                "index": index,
                "question_id": test_case.question_id,
                "question_type": test_case.question_type,
                "source_types": test_case.source_types,
                "query": test_case.query,
                "expected_doc_ids": test_case.expected_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "generated_answer": None,
                "generation_error": None,
                "ragas": {},
                "ragas_error": None,
                "ragas_cache_hit": False,
                "metrics": metrics,
                "skipped": skipped,
                "error": error,
                "results": [serialise_result(result) for result in retrieved_results],
            }
        )

        if enable_generation or enable_ragas:
            postprocess_jobs.append(
                {
                    "result_index": len(query_results) - 1,
                    "query_t0": query_t0,
                    "test_case": test_case,
                    "contexts": contexts,
                    "retrieved_doc_ids": retrieved_doc_ids,
                }
            )

        retrieval_progress.update(1)
        if retrieval_error_count:
            retrieval_progress.set_postfix(errors=retrieval_error_count)

    retrieval_progress.close()

    def apply_postprocess_update(
        result_index: int,
        update: dict[str, Any],
        query_t0: float,
        *,
        cache_hit: bool,
    ) -> None:
        query_result = query_results[result_index]
        query_result["generated_answer"] = update.get("generated_answer")
        query_result["generation_error"] = update.get("generation_error")
        query_result["ragas"] = update.get("ragas", {})
        query_result["ragas_error"] = update.get("ragas_error")
        query_result["ragas_cache_hit"] = cache_hit
        query_result["metrics"].update(update.get("metrics", {}))
        if cache_hit:
            if enable_generation or enable_ragas:
                query_result["metrics"]["generation_latency_ms"] = 0.0
            if enable_ragas:
                query_result["metrics"]["ragas_latency_ms"] = 0.0
        query_result["metrics"]["end_to_end_latency_ms"] = round(
            (time.monotonic() - query_t0) * 1000.0,
            1,
        )

    def run_postprocess_job(job: dict[str, Any]) -> tuple[int, dict[str, Any], bool]:
        test_case = job["test_case"]
        contexts = job["contexts"]
        retrieved_doc_ids = job["retrieved_doc_ids"]
        cache_key = _ragas_cache_key(
            mode=mode,
            test_case=test_case,
            retrieved_doc_ids=retrieved_doc_ids,
            contexts=contexts,
            ragas_metrics=ragas_metrics,
            max_context_chars=max_context_chars,
            namespace=ragas_cache_namespace,
        )

        if ragas_cache is not None:
            try:
                cached = ragas_cache.get(cache_key)
            except Exception:
                cached = None
            if cached is not None:
                return job["result_index"], cached, True

        update = _run_generation_and_ragas(
            query=test_case.query,
            contexts=contexts,
            enable_generation=enable_generation,
            enable_ragas=enable_ragas,
            answer_generator=answer_generator,
            ragas_evaluator=ragas_evaluator,
        )

        if ragas_cache is not None:
            can_cache_generation = bool(update.get("generated_answer")) and not update.get(
                "generation_error"
            )
            can_cache_ragas = (not enable_ragas) or (
                bool(update.get("ragas")) and not update.get("ragas_error")
            )
            if can_cache_generation and can_cache_ragas:
                try:
                    ragas_cache.set(cache_key, update)
                except Exception:
                    pass

        return job["result_index"], update, False

    if postprocess_jobs:
        worker_count = max(1, int(ragas_workers))
        postprocess_cache_hits = 0
        postprocess_error_count = 0
        postprocess_progress = _progress_bar(
            enabled=progress,
            total=len(postprocess_jobs),
            desc=f"RAGAS {mode}" if enable_ragas else f"Generation {mode}",
        )

        def update_postprocess_progress(
            *,
            cache_hit: bool,
            update: dict[str, Any],
        ) -> None:
            nonlocal postprocess_cache_hits, postprocess_error_count
            if cache_hit:
                postprocess_cache_hits += 1
            if update.get("generation_error") or update.get("ragas_error"):
                postprocess_error_count += 1
            postprocess_progress.update(1)
            postprocess_progress.set_postfix(
                cache=postprocess_cache_hits,
                errors=postprocess_error_count,
            )

        if worker_count == 1 or len(postprocess_jobs) == 1:
            for job in postprocess_jobs:
                result_index, update, cache_hit = run_postprocess_job(job)
                apply_postprocess_update(
                    result_index,
                    update,
                    job["query_t0"],
                    cache_hit=cache_hit,
                )
                update_postprocess_progress(cache_hit=cache_hit, update=update)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(run_postprocess_job, job): job
                    for job in postprocess_jobs
                }
                for future in as_completed(futures):
                    job = futures[future]
                    result_index, update, cache_hit = future.result()
                    apply_postprocess_update(
                        result_index,
                        update,
                        job["query_t0"],
                        cache_hit=cache_hit,
                    )
                    update_postprocess_progress(cache_hit=cache_hit, update=update)

        postprocess_progress.close()

    evaluated_count = sum(1 for item in query_results if not item.get("skipped"))
    error_count = sum(1 for item in query_results if item.get("error"))
    ragas_cache_hit_count = sum(
        1 for item in query_results if item.get("ragas_cache_hit")
    )

    return {
        "mode": mode,
        "query_count": len(query_results),
        "evaluated_query_count": evaluated_count,
        "skipped_query_count": len(query_results) - evaluated_count,
        "error_count": error_count,
        "ragas_cache_hit_count": ragas_cache_hit_count,
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

    ragas_metrics = [
        str(metric)
        for metric in report.get("ragas_metrics", [])
        if str(metric).strip()
    ]
    include_generation = bool(report.get("enable_generation")) or bool(
        report.get("enable_ragas")
    )
    include_ragas = bool(report.get("enable_ragas"))
    include_ragas_cache = bool(report.get("ragas_cache"))

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
    if include_generation:
        headers.append("Gen Latency ms")
    if include_ragas:
        headers.extend(ragas_metrics)
        headers.append("RAGAS Latency ms")
    if include_ragas_cache:
        headers.append("RAGAS Cache Hits")

    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for mode in report["modes"]:
        mode_result = report["results"][mode]
        aggregate = mode_result.get("aggregate_metrics", {})
        row = [
            mode,
            str(mode_result.get("evaluated_query_count", 0)),
            _format_metric(aggregate.get(f"recall@{top_k}")),
            _format_metric(aggregate.get(f"precision@{top_k}")),
            _format_metric(aggregate.get(f"mrr@{top_k}")),
            _format_metric(aggregate.get(f"ndcg@{top_k}")),
            _format_metric(aggregate.get(f"hit@{top_k}")),
            _format_metric(aggregate.get("latency_ms"), digits=1),
        ]
        if include_generation:
            row.append(_format_metric(aggregate.get("generation_latency_ms"), digits=1))
        if include_ragas:
            row.extend(_format_metric(aggregate.get(metric)) for metric in ragas_metrics)
            row.append(_format_metric(aggregate.get("ragas_latency_ms"), digits=1))
        if include_ragas_cache:
            row.append(str(mode_result.get("ragas_cache_hit_count", 0)))

        rows.append(
            "| "
            + " | ".join(row)
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
    if args.max_context_chars <= 0:
        raise ValueError("--max-context-chars must be greater than 0")
    if args.ragas_workers <= 0:
        raise ValueError("--ragas-workers must be greater than 0")

    config_path = _resolve_path(args.config)
    questions_path = _resolve_questions_file(args.questions_file)
    output_dir = _resolve_path(args.output_dir)
    ragas_cache_path = _resolve_path(args.ragas_cache) if args.ragas_cache else None
    effective_candidate_k = (
        max(args.candidate_k, args.top_k) if args.candidate_k > 0 else args.top_k * 2
    )
    enable_generation = bool(args.enable_generation or args.enable_ragas)
    ragas_metrics = [
        str(metric).strip().lower()
        for metric in args.ragas_metrics
        if str(metric).strip()
    ] or ["faithfulness"]

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

    answer_generator = _create_answer_generator(settings, enable_generation)
    ragas_evaluator = _create_ragas_evaluator(
        settings=settings,
        enabled=bool(args.enable_ragas),
        metrics=ragas_metrics,
    )
    ragas_cache = RagasEvalCache(ragas_cache_path) if ragas_cache_path else None
    ragas_cache_namespace = {
        "llm_provider": getattr(settings.llm, "provider", None),
        "llm_model": getattr(settings.llm, "model", None),
        "embedding_provider": getattr(settings.embedding, "provider", None),
        "embedding_model": getattr(settings.embedding, "model", None),
    }

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
        "enable_generation": enable_generation,
        "enable_ragas": bool(args.enable_ragas),
        "ragas_metrics": ragas_metrics,
        "ragas_workers": int(args.ragas_workers),
        "ragas_cache": str(ragas_cache_path) if ragas_cache_path else None,
        "ragas_cache_namespace": ragas_cache_namespace if ragas_cache_path else None,
        "progress": bool(args.progress),
        "max_context_chars": args.max_context_chars,
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
            enable_generation=enable_generation,
            enable_ragas=bool(args.enable_ragas),
            answer_generator=answer_generator,
            ragas_evaluator=ragas_evaluator,
            max_context_chars=args.max_context_chars,
            ragas_workers=args.ragas_workers,
            ragas_cache=ragas_cache,
            ragas_metrics=ragas_metrics,
            ragas_cache_namespace=ragas_cache_namespace,
            progress=bool(args.progress),
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
