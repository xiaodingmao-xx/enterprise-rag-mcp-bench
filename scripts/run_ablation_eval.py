#!/usr/bin/env python
"""Run retrieval ablation evaluation across search modes.

Modes:
    dense          Dense vector retrieval only
    bm25           Sparse BM25 retrieval only
    hybrid         Dense + BM25 with RRF fusion
    hybrid_rerank  Hybrid retrieval followed by configured reranker

Results are written to eval/results/{timestamp}.json by default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.hybrid_search import create_hybrid_search
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import create_core_reranker
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.core.settings import load_settings
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.evaluation.eval_runner import GoldenTestCase, load_test_set
from src.observability.evaluation.ir_metrics import evaluate_ranking_at_k


MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation evaluation for dense, BM25, hybrid, and reranked hybrid search.",
    )
    parser.add_argument(
        "--dataset",
        "--test-set",
        dest="dataset",
        default="tests/fixtures/golden_test_set.json",
        help="Golden test set JSON path.",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Settings YAML path.",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Collection name to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="K for Recall@K, Precision@K, MRR@K, and NDCG@K.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=0,
        help="Candidate count before rerank. Defaults to 2 * top-k for hybrid_rerank.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        help="Modes to run. Defaults to all modes.",
    )
    parser.add_argument(
        "--mode",
        dest="mode_repeats",
        action="append",
        choices=MODES,
        help="Single mode to run. Can be repeated.",
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


def _select_modes(args: argparse.Namespace) -> list[str]:
    modes: list[str] = []
    if args.modes:
        modes.extend(args.modes)
    if args.mode_repeats:
        modes.extend(args.mode_repeats)
    if not modes:
        modes = list(MODES)

    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def _build_components(settings: Any, collection: str) -> dict[str, Any]:
    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding_client = EmbeddingFactory.create(settings)

    dense_retriever = create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )

    bm25_indexer = BM25Indexer(index_dir=str(PROJECT_ROOT / "data" / "db" / "bm25" / collection))
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection

    query_processor = QueryProcessor()
    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    return {
        "dense_retriever": dense_retriever,
        "sparse_retriever": sparse_retriever,
        "query_processor": query_processor,
        "hybrid_search": hybrid_search,
        "reranker": create_core_reranker(settings=settings),
        "collection": collection,
    }


def _run_retrieval(
    mode: str,
    query: str,
    top_k: int,
    candidate_k: int,
    components: dict[str, Any],
) -> list[Any]:
    if mode == "dense":
        return components["dense_retriever"].retrieve(query=query, top_k=top_k)

    if mode == "bm25":
        processed = components["query_processor"].process(query)
        if not processed.keywords:
            return []
        return components["sparse_retriever"].retrieve(
            keywords=processed.keywords,
            top_k=top_k,
            collection=components["collection"],
        )

    if mode == "hybrid":
        results = components["hybrid_search"].search(query=query, top_k=top_k)
        return results if isinstance(results, list) else results.results

    if mode == "hybrid_rerank":
        effective_candidate_k = candidate_k if candidate_k > 0 else top_k * 2
        effective_candidate_k = max(effective_candidate_k, top_k)
        candidates = components["hybrid_search"].search(
            query=query,
            top_k=effective_candidate_k,
        )
        candidates = candidates if isinstance(candidates, list) else candidates.results
        if not candidates:
            return []
        rerank_result = components["reranker"].rerank(
            query=query,
            results=candidates,
            top_k=top_k,
        )
        return rerank_result.results

    raise ValueError(f"Unsupported ablation mode: {mode}")


def _chunk_ids(results: Sequence[Any]) -> list[str]:
    ids: list[str] = []
    for result in results:
        if isinstance(result, dict):
            chunk_id = result.get("chunk_id") or result.get("id") or ""
        else:
            chunk_id = getattr(result, "chunk_id", getattr(result, "id", ""))
        ids.append(str(chunk_id))
    return ids


def _result_metadata(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        metadata = result.get("metadata", {})
    else:
        metadata = getattr(result, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _normalise_path(value: str) -> str:
    return str(value).strip().replace("\\", "/").lower()


def _basename(value: str) -> str:
    return PurePosixPath(_normalise_path(value)).name


def _expected_sources_use_basename(expected_sources: Iterable[str]) -> bool:
    normalised = [_normalise_path(item) for item in expected_sources if str(item).strip()]
    return bool(normalised) and all("/" not in item for item in normalised)


def _source_labels(results: Sequence[Any], expected_sources: Sequence[str]) -> list[str]:
    use_basename = _expected_sources_use_basename(expected_sources)
    labels: list[str] = []
    for result in results:
        metadata = _result_metadata(result)
        source = (
            metadata.get("source_path")
            or metadata.get("source")
            or metadata.get("file_path")
            or metadata.get("source_file")
            or ""
        )
        source_text = str(source)
        labels.append(_basename(source_text) if use_basename else _normalise_path(source_text))
    return labels


def _relevance_labels(test_case: GoldenTestCase) -> tuple[str, list[str]]:
    if test_case.expected_chunk_ids:
        return "chunk_id", [str(item) for item in test_case.expected_chunk_ids]

    if test_case.expected_sources:
        if _expected_sources_use_basename(test_case.expected_sources):
            return "source", [_basename(item) for item in test_case.expected_sources]
        return "source", [_normalise_path(item) for item in test_case.expected_sources]

    return "none", []


def _serialise_result(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        data = result.to_dict()
    elif isinstance(result, dict):
        data = dict(result)
    else:
        data = {
            "chunk_id": getattr(result, "chunk_id", ""),
            "score": getattr(result, "score", 0.0),
            "text": getattr(result, "text", ""),
            "metadata": getattr(result, "metadata", {}),
        }

    text = str(data.get("text", ""))
    data["text_preview"] = text.replace("\n", " ")[:240]
    data.pop("text", None)
    return data


def _aggregate_query_metrics(query_results: Sequence[dict[str, Any]]) -> dict[str, float]:
    metric_values: dict[str, list[float]] = {}
    for item in query_results:
        if item.get("skipped"):
            continue
        for key, value in item.get("metrics", {}).items():
            metric_values.setdefault(key, []).append(float(value))

    return {
        key: sum(values) / len(values)
        for key, values in sorted(metric_values.items())
        if values
    }


def _evaluate_mode(
    mode: str,
    test_cases: Sequence[GoldenTestCase],
    top_k: int,
    candidate_k: int,
    components: dict[str, Any],
) -> dict[str, Any]:
    mode_t0 = time.monotonic()
    query_results: list[dict[str, Any]] = []

    for index, test_case in enumerate(test_cases, start=1):
        query_t0 = time.monotonic()
        relevance_type, relevant_labels = _relevance_labels(test_case)
        skipped = not relevant_labels
        error: str | None = None
        retrieved_results: list[Any] = []

        if not skipped:
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

        retrieved_ids = _chunk_ids(retrieved_results)
        retrieved_sources = _source_labels(
            retrieved_results,
            test_case.expected_sources,
        )
        retrieved_labels = (
            retrieved_ids if relevance_type == "chunk_id" else retrieved_sources
        )
        metrics = (
            {}
            if skipped
            else evaluate_ranking_at_k(retrieved_labels, relevant_labels, top_k)
        )

        query_results.append(
            {
                "index": index,
                "query": test_case.query,
                "relevance_type": relevance_type,
                "relevant_labels": relevant_labels,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_sources": retrieved_sources,
                "metrics": metrics,
                "elapsed_ms": round((time.monotonic() - query_t0) * 1000.0, 1),
                "skipped": skipped,
                "error": error,
                "results": [_serialise_result(result) for result in retrieved_results],
            }
        )

    evaluated_count = sum(1 for item in query_results if not item["skipped"])
    skipped_count = len(query_results) - evaluated_count

    return {
        "mode": mode,
        "query_count": len(query_results),
        "evaluated_query_count": evaluated_count,
        "skipped_query_count": skipped_count,
        "aggregate_metrics": _aggregate_query_metrics(query_results),
        "elapsed_ms": round((time.monotonic() - mode_t0) * 1000.0, 1),
        "query_results": query_results,
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_markdown_table(report: dict[str, Any], top_k: int) -> str:
    headers = [
        "Mode",
        "Evaluated",
        f"Recall@{top_k}",
        f"Precision@{top_k}",
        f"MRR@{top_k}",
        f"NDCG@{top_k}",
        "Elapsed ms",
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
                    _format_metric(mode_result.get("elapsed_ms")),
                ]
            )
            + " |"
        )

    return "\n".join(rows)


def run_ablation_eval(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path | None]:
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")
    if args.candidate_k < 0:
        raise ValueError("--candidate-k must be greater than or equal to 0")

    config_path = _resolve_path(args.config)
    dataset_path = _resolve_path(args.dataset)
    output_dir = _resolve_path(args.output_dir)

    settings = load_settings(config_path)
    test_cases = load_test_set(dataset_path)
    modes = _select_modes(args)
    components = _build_components(settings, args.collection)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{timestamp}.json"

    report: dict[str, Any] = {
        "timestamp": timestamp,
        "dataset": str(dataset_path),
        "config": str(config_path),
        "collection": args.collection,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k if args.candidate_k > 0 else args.top_k * 2,
        "modes": modes,
        "test_case_count": len(test_cases),
        "results": {},
    }

    total_t0 = time.monotonic()
    for mode in modes:
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
        report, json_path, markdown_path = run_ablation_eval(args)
    except Exception as exc:
        print(f"[FAIL] Ablation evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Ablation results saved: {json_path}")
    if markdown_path is not None:
        print(f"[OK] Markdown table saved: {markdown_path}")
    print(f"[OK] Modes: {', '.join(report['modes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
