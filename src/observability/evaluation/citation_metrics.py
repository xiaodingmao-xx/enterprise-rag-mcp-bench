"""Citation reliability metrics for MMDocRAG evaluation."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Iterable, Sequence


def _normalise_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").lower()


def _basename(value: Any) -> str:
    return PurePosixPath(_normalise_path(value)).name


def _metadata_from_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        metadata = result.get("metadata", {})
    else:
        metadata = getattr(result, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _chunk_id_from_result(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("chunk_id") or result.get("id") or "")
    return str(getattr(result, "chunk_id", getattr(result, "id", "")))


def _add_source_labels(labels: set[str], value: Any) -> None:
    normalised = _normalise_path(value)
    if not normalised:
        return
    labels.add(f"source:{normalised}")
    labels.add(f"source:{_basename(normalised)}")


def _add_page_label(labels: set[str], value: Any) -> None:
    if value is None or str(value).strip() == "":
        return
    labels.add(f"page:{str(value).strip()}")


def _labels_from_metadata(result: Any) -> set[str]:
    labels: set[str] = set()
    metadata = _metadata_from_result(result)

    chunk_id = _chunk_id_from_result(result)
    if chunk_id:
        labels.add(f"chunk_id:{chunk_id}")

    source = (
        metadata.get("source_path")
        or metadata.get("source")
        or metadata.get("file_path")
        or metadata.get("source_file")
    )
    if source:
        _add_source_labels(labels, source)

    for key in ("page", "page_num", "page_number"):
        if key in metadata:
            _add_page_label(labels, metadata.get(key))

    for key in ("image_id", "table_id"):
        if metadata.get(key):
            labels.add(f"{key}:{metadata[key]}")

    for key, label_name in (
        ("image_refs", "image_id"),
        ("images", "image_id"),
        ("table_refs", "table_id"),
        ("tables", "table_id"),
    ):
        values = metadata.get(key)
        if isinstance(values, (list, tuple, set)):
            for item in values:
                if isinstance(item, dict):
                    identifier = item.get("id") or item.get(label_name)
                else:
                    identifier = item
                if identifier:
                    labels.add(f"{label_name}:{identifier}")

    return {label.lower() for label in labels}


def expected_citation_labels(
    *,
    expected_sources: Iterable[str] | None = None,
    expected_pages: Iterable[Any] | None = None,
    expected_chunk_ids: Iterable[str] | None = None,
    expected_evidence: Sequence[dict[str, Any]] | None = None,
) -> set[str]:
    """Build comparable citation labels from optional golden annotations."""

    labels: set[str] = set()

    for chunk_id in expected_chunk_ids or []:
        if str(chunk_id).strip():
            labels.add(f"chunk_id:{str(chunk_id).strip()}")

    for source in expected_sources or []:
        _add_source_labels(labels, source)

    for page in expected_pages or []:
        _add_page_label(labels, page)

    for evidence in expected_evidence or []:
        if not isinstance(evidence, dict):
            continue
        if evidence.get("chunk_id"):
            labels.add(f"chunk_id:{evidence['chunk_id']}")
        if evidence.get("source"):
            _add_source_labels(labels, evidence["source"])
        if evidence.get("page") is not None:
            _add_page_label(labels, evidence["page"])
        if evidence.get("image_id"):
            labels.add(f"image_id:{evidence['image_id']}")
        if evidence.get("table_id"):
            labels.add(f"table_id:{evidence['table_id']}")

    return {label.lower() for label in labels}


def _answer_mentions_label(answer: str, label: str) -> bool:
    kind, _, value = label.partition(":")
    if not value:
        return False

    lowered = answer.lower()
    if kind == "page":
        escaped = re.escape(value)
        patterns = [
            rf"\bpage\s*{escaped}\b",
            rf"\bp\.\s*{escaped}\b",
            rf"第\s*{escaped}\s*页",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    return value.lower() in lowered


def answer_citation_labels(
    generated_answer: str | None,
    retrieved_results: Sequence[Any] | None = None,
) -> set[str]:
    """Extract citation labels from answer text and bracketed result indexes."""

    if not generated_answer or not generated_answer.strip():
        return set()

    answer = generated_answer.strip()
    labels: set[str] = set()
    retrieved = list(retrieved_results or [])
    retrieved_label_sets = [_labels_from_metadata(result) for result in retrieved]

    for key in ("chunk_id", "image_id", "table_id"):
        for match in re.finditer(
            rf"{key}\s*[:=]\s*([A-Za-z0-9_.\-]+)",
            answer,
            flags=re.IGNORECASE,
        ):
            labels.add(f"{key}:{match.group(1)}")

    for match in re.finditer(r"\[(\d+)\]", answer):
        index = int(match.group(1)) - 1
        if 0 <= index < len(retrieved_label_sets):
            labels.update(retrieved_label_sets[index])

    for result_labels in retrieved_label_sets:
        for label in result_labels:
            if _answer_mentions_label(answer, label):
                labels.add(label)

    return {label.lower() for label in labels}


def citation_accuracy(
    *,
    generated_answer: str | None,
    retrieved_results: Sequence[Any] | None = None,
    expected_sources: Iterable[str] | None = None,
    expected_pages: Iterable[Any] | None = None,
    expected_chunk_ids: Iterable[str] | None = None,
    expected_evidence: Sequence[dict[str, Any]] | None = None,
) -> float | None:
    """Compute exact-label citation coverage for a generated answer.

    Returns ``None`` when there is no citation annotation in the golden sample.
    """

    expected = expected_citation_labels(
        expected_sources=expected_sources,
        expected_pages=expected_pages,
        expected_chunk_ids=expected_chunk_ids,
        expected_evidence=expected_evidence,
    )
    if not expected:
        return None

    observed = answer_citation_labels(generated_answer, retrieved_results)
    if not observed:
        return 0.0

    hits = expected.intersection(observed)
    return len(hits) / len(expected)
