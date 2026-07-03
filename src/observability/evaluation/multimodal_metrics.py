"""Multimodal retrieval metrics for MMDocRAG evaluation."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


IMAGE_MODALITIES = {"image", "figure", "chart", "diagram"}
TABLE_MODALITIES = {"table"}


def _normalise_modality(value: Any) -> set[str]:
    """Normalise modality metadata values to a lowercase set."""

    if value is None:
        return set()

    if isinstance(value, (list, tuple, set)):
        modalities: set[str] = set()
        for item in value:
            modalities.update(_normalise_modality(item))
        return modalities

    text = str(value).strip().lower()
    if not text:
        return set()

    aliases = {
        "img": "image",
        "picture": "image",
        "photo": "image",
        "graph": "chart",
        "plot": "chart",
        "tabular": "table",
    }
    canonical = aliases.get(text, text)

    if "image" in canonical or "figure" in canonical:
        return {"image"}
    if "table" in canonical:
        return {"table"}
    if "chart" in canonical or "diagram" in canonical:
        return {canonical if canonical in {"chart", "diagram"} else "chart"}
    if "text" in canonical:
        return {"text"}
    return {canonical}


def _metadata_from_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        metadata = result.get("metadata", {})
    else:
        metadata = getattr(result, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def extract_modalities_from_metadata(metadata: dict[str, Any]) -> set[str]:
    """Infer content modalities from a retrieval result metadata dictionary."""

    modalities: set[str] = set()
    modalities.update(_normalise_modality(metadata.get("modality")))
    modalities.update(_normalise_modality(metadata.get("content_type")))

    if _is_truthy(metadata.get("has_image")) or _is_truthy(metadata.get("image_id")):
        modalities.add("image")
    if _is_truthy(metadata.get("has_table")) or _is_truthy(metadata.get("table_id")):
        modalities.add("table")
    if _is_truthy(metadata.get("image_refs")) or _is_truthy(metadata.get("images")):
        modalities.add("image")
    if _is_truthy(metadata.get("table_refs")) or _is_truthy(metadata.get("tables")):
        modalities.add("table")

    return modalities


def extract_modalities_from_results(
    retrieved_results: Sequence[Any],
    k: int,
) -> set[str]:
    """Return modalities covered by the top-k retrieval results."""

    if k <= 0:
        return set()

    modalities: set[str] = set()
    for result in retrieved_results[:k]:
        modalities.update(extract_modalities_from_metadata(_metadata_from_result(result)))
    return modalities


def _normalise_expected_modalities(
    expected_modalities: Iterable[str] | str | None,
) -> set[str]:
    if expected_modalities is None:
        return set()
    if isinstance(expected_modalities, str):
        return _normalise_modality(expected_modalities)
    return _normalise_modality(list(expected_modalities))


def modality_recall_at_k(
    retrieved_results: Sequence[Any],
    expected_modalities: Iterable[str] | str | None,
    k: int,
) -> float | None:
    """Compute the fraction of expected evidence modalities covered in top-k.

    Returns ``None`` when the sample has no ``expected_modalities`` annotation.
    Callers should skip ``None`` during aggregate averaging.
    """

    expected = _normalise_expected_modalities(expected_modalities)
    if not expected:
        return None

    retrieved = extract_modalities_from_results(retrieved_results, k)
    hits = len(expected.intersection(retrieved))
    return hits / len(expected)


def image_hit_at_k(
    retrieved_results: Sequence[Any],
    expected_modalities: Iterable[str] | str | None,
    k: int,
) -> float | None:
    """Return 1 when an image-like modality is required and found in top-k."""

    expected = _normalise_expected_modalities(expected_modalities)
    if not expected.intersection(IMAGE_MODALITIES):
        return None

    retrieved = extract_modalities_from_results(retrieved_results, k)
    return 1.0 if retrieved.intersection(IMAGE_MODALITIES) else 0.0


def table_hit_at_k(
    retrieved_results: Sequence[Any],
    expected_modalities: Iterable[str] | str | None,
    k: int,
) -> float | None:
    """Return 1 when table evidence is required and found in top-k."""

    expected = _normalise_expected_modalities(expected_modalities)
    if not expected.intersection(TABLE_MODALITIES):
        return None

    retrieved = extract_modalities_from_results(retrieved_results, k)
    return 1.0 if retrieved.intersection(TABLE_MODALITIES) else 0.0


def evaluate_multimodal_at_k(
    retrieved_results: Sequence[Any],
    expected_modalities: Iterable[str] | str | None,
    k: int,
) -> dict[str, float]:
    """Compute the MMDocRAG multimodal metric bundle for one query."""

    values = {
        f"modality_recall@{k}": modality_recall_at_k(
            retrieved_results,
            expected_modalities,
            k,
        ),
        f"image_hit@{k}": image_hit_at_k(retrieved_results, expected_modalities, k),
        f"table_hit@{k}": table_hit_at_k(retrieved_results, expected_modalities, k),
    }
    return {key: value for key, value in values.items() if value is not None}
