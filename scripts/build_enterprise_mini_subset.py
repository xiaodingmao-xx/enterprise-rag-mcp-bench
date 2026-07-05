#!/usr/bin/env python
"""Build a small EnterpriseRAG-Bench subset for retrieval evaluation."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DSID_PATTERN = re.compile(r"(dsid_[0-9A-Za-z]+)", re.IGNORECASE)
DEFAULT_SOURCE_TYPES = (
    "confluence",
    "github",
    "jira",
    "linear",
    "google_drive",
    "slack",
    "gmail",
)
DEFAULT_QUESTION_WEIGHTS = {
    "basic": 15,
    "semantic": 15,
    "constrained": 10,
    "intra_document_reasoning": 10,
}
SUPPORTED_DOCUMENT_SUFFIXES = {".txt", ".md", ".json", ".jsonl"}
DOC_ID_FIELDS = (
    "doc_id",
    "document_id",
    "dsid",
    "id",
    "source_id",
)
TITLE_FIELDS = ("title", "name", "subject", "heading")
CONTENT_FIELDS = (
    "content",
    "text",
    "body",
    "markdown",
    "page_content",
    "description",
    "message",
)


@dataclass(frozen=True)
class QuestionCandidate:
    record: dict[str, Any]
    row_index: int
    question_type: str
    source_types: tuple[str, ...]
    expected_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class DocumentCandidate:
    doc_id: str
    source_type: str
    input_path: Path
    record: dict[str, Any] | None = None
    record_index: int | None = None
    title: str | None = None
    content: str | None = None
    original_format: str = "text"


@dataclass
class DocumentIndex:
    by_id: dict[str, DocumentCandidate]
    ids_by_source: dict[str, list[str]]
    duplicate_doc_ids: list[dict[str, str]]
    duplicate_doc_id_count: int
    scanned_files: int
    scanned_records: int
    skipped_files: int
    skipped_records: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an EnterpriseRAG-Bench Mini subset from questions and documents.",
    )
    parser.add_argument(
        "--questions-file",
        required=True,
        help=(
            "Path to questions.jsonl/.json/.parquet, all_documents, or the "
            "EnterpriseRAG-Bench root directory."
        ),
    )
    parser.add_argument(
        "--documents-root",
        required=True,
        help=(
            "Path to exported documents, all_documents, generated_data/sources, "
            "or the EnterpriseRAG-Bench root directory."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory for the mini subset.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=50,
        help="Maximum question count to sample. Defaults to 50.",
    )
    parser.add_argument(
        "--target-docs",
        type=int,
        default=20000,
        help="Target document count. Gold documents are always kept when found.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=None,
        help=(
            "Question types to include. Defaults to Basic, Semantic, Constrained, "
            "and Intra-Document Reasoning with 15/15/10/10 weighting."
        ),
    )
    parser.add_argument(
        "--source-types",
        nargs="+",
        default=list(DEFAULT_SOURCE_TYPES),
        help="Source types to include.",
    )
    return parser.parse_args(argv)


def configure_stdio() -> None:
    if sys.platform != "win32":
        return
    if "pytest" in Path(sys.argv[0]).name.lower():
        return

    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def extract_doc_id(value: Any) -> str | None:
    """Extract a normalized EnterpriseRAG-Bench dsid from a value."""

    if value is None:
        return None

    if isinstance(value, dict):
        for field in DOC_ID_FIELDS:
            doc_id = extract_doc_id(value.get(field))
            if doc_id:
                return doc_id
        for nested_value in value.values():
            doc_id = extract_doc_id(nested_value)
            if doc_id:
                return doc_id
        return None

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):  # noqa: UP038
        for nested_value in value:
            doc_id = extract_doc_id(nested_value)
            if doc_id:
                return doc_id
        return None

    match = DSID_PATTERN.search(str(value))
    if match is None:
        return None
    return match.group(1).lower()


def normalise_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []

    if hasattr(value, "tolist"):
        return coerce_str_list(value.tolist())

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    return coerce_str_list(parser(stripped))
                except Exception:
                    continue
        return [stripped]

    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):  # noqa: UP038
        return [str(item).strip() for item in value if str(item).strip()]

    return [str(value).strip()] if str(value).strip() else []


def normalise_expected_doc_ids(value: Any) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for item in coerce_str_list(value):
        doc_id = extract_doc_id(item) or item.strip().lower()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids


def resolve_project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def resolve_questions_file(path: str | Path) -> Path:
    path_obj = resolve_project_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Questions path not found: {path_obj}")

    if path_obj.is_file():
        return path_obj

    candidates = [
        path_obj / "questions.jsonl",
        path_obj / "all_documents" / "questions.jsonl",
        path_obj / "data" / "questions" / "test.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find questions.jsonl or data/questions/test.parquet under "
        f"{path_obj}"
    )


def resolve_documents_root(path: str | Path) -> Path:
    path_obj = resolve_project_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Documents root not found: {path_obj}")
    if (path_obj / "all_documents").is_dir():
        return path_obj / "all_documents"
    if (path_obj / "generated_data" / "sources").is_dir():
        return path_obj / "generated_data" / "sources"
    return path_obj


def iter_question_records(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    log_warn(f"Skipping bad JSON at {path}:{line_number}: {exc.msg}")
                    continue
                if isinstance(record, dict):
                    yield record
                else:
                    log_warn(f"Skipping non-object JSONL row at {path}:{line_number}")
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        records = data.get("questions", data) if isinstance(data, dict) else data
        if not isinstance(records, list):
            raise ValueError("Questions JSON must be a list or {'questions': [...]}")
        for record in records:
            if isinstance(record, dict):
                yield record
        return

    if suffix == ".parquet":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet questions requires the optional 'datasets' package."
            ) from exc
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        for record in dataset:
            yield dict(record)
        return

    raise ValueError(f"Unsupported questions file type: {path.suffix}")


def allocate_counts(total: int, weights: dict[str, int]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("--max-questions must be greater than 0")
    if not weights:
        raise ValueError("At least one question type is required")

    weight_sum = sum(weights.values())
    raw_counts: list[tuple[str, int, float]] = []
    allocated = 0
    for key, weight in weights.items():
        exact = total * (weight / weight_sum)
        floor = int(exact)
        allocated += floor
        raw_counts.append((key, floor, exact - floor))

    remainder = total - allocated
    raw_counts.sort(key=lambda item: item[2], reverse=True)
    counts = {key: floor for key, floor, _ in raw_counts}
    for key, _, _ in raw_counts[:remainder]:
        counts[key] += 1
    return counts


def build_question_candidates(
    records: Iterable[dict[str, Any]],
    *,
    question_types: set[str],
    source_types: set[str],
) -> list[QuestionCandidate]:
    candidates: list[QuestionCandidate] = []
    for row_index, record in enumerate(records, start=1):
        question_type = normalise_label(record.get("question_type"))
        if question_type not in question_types:
            continue

        current_source_types = tuple(normalise_label(item) for item in coerce_str_list(record.get("source_types")))
        if source_types and not set(current_source_types).intersection(source_types):
            continue

        expected_doc_ids = tuple(normalise_expected_doc_ids(record.get("expected_doc_ids")))
        if not expected_doc_ids:
            continue

        question = str(record.get("question") or record.get("query") or "").strip()
        if not question:
            continue

        candidates.append(
            QuestionCandidate(
                record=dict(record),
                row_index=row_index,
                question_type=question_type,
                source_types=current_source_types,
                expected_doc_ids=expected_doc_ids,
            )
        )
    return candidates


def source_balance_score(candidate: QuestionCandidate, source_counts: Counter[str]) -> int:
    if not candidate.source_types:
        return 0
    return sum(source_counts[source_type] for source_type in candidate.source_types)


def pick_balanced_questions(
    pool: list[QuestionCandidate],
    count: int,
    *,
    rng: random.Random,
    selected_rows: set[int],
    source_counts: Counter[str],
) -> list[QuestionCandidate]:
    remaining = [candidate for candidate in pool if candidate.row_index not in selected_rows]
    rng.shuffle(remaining)
    picked: list[QuestionCandidate] = []

    while remaining and len(picked) < count:
        best_index = min(
            range(len(remaining)),
            key=lambda index: source_balance_score(remaining[index], source_counts),
        )
        candidate = remaining.pop(best_index)
        picked.append(candidate)
        selected_rows.add(candidate.row_index)
        for source_type in candidate.source_types:
            source_counts[source_type] += 1

    return picked


def select_questions(
    questions_path: Path,
    *,
    max_questions: int,
    requested_question_types: list[str] | None,
    source_types: set[str],
    rng: random.Random,
) -> list[QuestionCandidate]:
    if requested_question_types:
        weights = {normalise_label(item): 1 for item in requested_question_types}
    else:
        weights = dict(DEFAULT_QUESTION_WEIGHTS)

    question_types = set(weights)
    candidates = build_question_candidates(
        iter_question_records(questions_path),
        question_types=question_types,
        source_types=source_types,
    )
    if not candidates:
        raise ValueError("No eligible questions found after filtering")

    quotas = allocate_counts(max_questions, weights)
    by_type: dict[str, list[QuestionCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_type[candidate.question_type].append(candidate)

    selected: list[QuestionCandidate] = []
    selected_rows: set[int] = set()
    source_counts: Counter[str] = Counter()

    for question_type, quota in quotas.items():
        selected.extend(
            pick_balanced_questions(
                by_type.get(question_type, []),
                quota,
                rng=rng,
                selected_rows=selected_rows,
                source_counts=source_counts,
            )
        )

    if len(selected) < max_questions:
        selected.extend(
            pick_balanced_questions(
                candidates,
                max_questions - len(selected),
                rng=rng,
                selected_rows=selected_rows,
                source_counts=source_counts,
            )
        )

    return selected[:max_questions]


def infer_source_type(path: Path, root: Path, allowed_source_types: set[str]) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts

    for part in parts[:-1]:
        label = normalise_label(part)
        if label in allowed_source_types:
            return label

    return normalise_label(parts[0]) if parts else "unknown"


def read_text_prefix(path: Path, limit: int = 8192) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def value_from_fields(record: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, (dict, list)):  # noqa: UP038
            return json.dumps(value, ensure_ascii=False)
        text = str(value).strip()
        if text:
            return text
    return None


def iter_json_document_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield line_number, record
        return

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return

    if isinstance(data, dict):
        for key in ("documents", "records", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                for index, record in enumerate(value, start=1):
                    if isinstance(record, dict):
                        yield index, record
                return
        yield 1, data
        return

    if isinstance(data, list):
        for index, record in enumerate(data, start=1):
            if isinstance(record, dict):
                yield index, record


def document_candidate_from_record(
    path: Path,
    record_index: int,
    record: dict[str, Any],
    *,
    fallback_source_type: str,
) -> DocumentCandidate | None:
    doc_id = extract_doc_id({field: record.get(field) for field in DOC_ID_FIELDS})
    if doc_id is None:
        doc_id = extract_doc_id(record) or extract_doc_id(path.name) or extract_doc_id(str(path))
    if doc_id is None:
        return None

    source_type = normalise_label(record.get("source_type") or fallback_source_type)
    title = value_from_fields(record, TITLE_FIELDS)
    content = value_from_fields(record, CONTENT_FIELDS)
    if not content:
        content = json.dumps(record, ensure_ascii=False, indent=2)

    return DocumentCandidate(
        doc_id=doc_id,
        source_type=source_type,
        input_path=path,
        record=dict(record),
        record_index=record_index,
        title=title,
        content=content,
        original_format=path.suffix.lower().lstrip(".") or "json",
    )


def add_document_candidate(
    candidate: DocumentCandidate,
    *,
    by_id: dict[str, DocumentCandidate],
    ids_by_source: dict[str, list[str]],
    duplicate_doc_ids: list[dict[str, str]],
) -> int:
    existing = by_id.get(candidate.doc_id)
    if existing is not None:
        if len(duplicate_doc_ids) < 1000:
            duplicate_doc_ids.append(
                {
                    "doc_id": candidate.doc_id,
                    "kept_path": str(existing.input_path),
                    "duplicate_path": str(candidate.input_path),
                }
            )
        return 1

    by_id[candidate.doc_id] = candidate
    ids_by_source[candidate.source_type].append(candidate.doc_id)
    return 0


def build_document_index(
    documents_root: Path,
    *,
    source_types: set[str],
) -> DocumentIndex:
    by_id: dict[str, DocumentCandidate] = {}
    ids_by_source: dict[str, list[str]] = defaultdict(list)
    duplicate_doc_ids: list[dict[str, str]] = []
    duplicate_count = 0
    scanned_files = 0
    scanned_records = 0
    skipped_files = 0
    skipped_records = 0

    for path in documents_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.lower().startswith("questions."):
            continue

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
            continue

        scanned_files += 1
        fallback_source_type = infer_source_type(path, documents_root, source_types)

        if suffix in {".json", ".jsonl"}:
            for record_index, record in iter_json_document_records(path):
                scanned_records += 1
                candidate = document_candidate_from_record(
                    path,
                    record_index,
                    record,
                    fallback_source_type=fallback_source_type,
                )
                if candidate is None:
                    skipped_records += 1
                    continue
                if source_types and candidate.source_type not in source_types:
                    skipped_records += 1
                    continue
                duplicate_count += add_document_candidate(
                    candidate,
                    by_id=by_id,
                    ids_by_source=ids_by_source,
                    duplicate_doc_ids=duplicate_doc_ids,
                )
            continue

        if source_types and fallback_source_type not in source_types:
            skipped_files += 1
            continue

        doc_id = extract_doc_id(path.name) or extract_doc_id(str(path))
        if doc_id is None:
            doc_id = extract_doc_id(read_text_prefix(path))
        if doc_id is None:
            skipped_files += 1
            continue

        candidate = DocumentCandidate(
            doc_id=doc_id,
            source_type=fallback_source_type,
            input_path=path,
            original_format=suffix.lstrip(".") or "text",
        )
        duplicate_count += add_document_candidate(
            candidate,
            by_id=by_id,
            ids_by_source=ids_by_source,
            duplicate_doc_ids=duplicate_doc_ids,
        )

    return DocumentIndex(
        by_id=by_id,
        ids_by_source=dict(ids_by_source),
        duplicate_doc_ids=duplicate_doc_ids,
        duplicate_doc_id_count=duplicate_count,
        scanned_files=scanned_files,
        scanned_records=scanned_records,
        skipped_files=skipped_files,
        skipped_records=skipped_records,
    )


def collect_gold_doc_requirements(
    selected_questions: list[QuestionCandidate],
) -> dict[str, dict[str, Any]]:
    requirements: dict[str, dict[str, Any]] = {}
    for candidate in selected_questions:
        question_id = str(
            candidate.record.get("question_id")
            or candidate.record.get("id")
            or candidate.row_index
        )
        for doc_id in candidate.expected_doc_ids:
            item = requirements.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "question_ids": [],
                    "source_types": set(),
                },
            )
            item["question_ids"].append(question_id)
            item["source_types"].update(candidate.source_types)

    for item in requirements.values():
        item["source_types"] = sorted(item["source_types"])
    return requirements


def draw_from_sources(
    sources: Iterable[str],
    *,
    count: int,
    document_index: DocumentIndex,
    selected_doc_ids: set[str],
    rng: random.Random,
) -> list[str]:
    source_order = [source for source in dict.fromkeys(sources) if source in document_index.ids_by_source]
    rng.shuffle(source_order)
    pools: dict[str, list[str]] = {}
    for source in source_order:
        ids = [doc_id for doc_id in document_index.ids_by_source[source] if doc_id not in selected_doc_ids]
        rng.shuffle(ids)
        pools[source] = ids

    drawn: list[str] = []
    while len(drawn) < count and any(pools.values()):
        for source in list(source_order):
            if len(drawn) >= count:
                break
            pool = pools.get(source) or []
            while pool and pool[-1] in selected_doc_ids:
                pool.pop()
            if not pool:
                continue
            doc_id = pool.pop()
            selected_doc_ids.add(doc_id)
            drawn.append(doc_id)
    return drawn


def select_documents(
    selected_questions: list[QuestionCandidate],
    document_index: DocumentIndex,
    *,
    target_docs: int,
    source_types: set[str],
    rng: random.Random,
) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    if target_docs <= 0:
        raise ValueError("--target-docs must be greater than 0")

    requirements = collect_gold_doc_requirements(selected_questions)
    selected_doc_ids: set[str] = set()
    ordered_doc_ids: list[str] = []
    missing_gold_docs: list[dict[str, Any]] = []
    gold_source_types: list[str] = []
    found_gold_doc_ids: set[str] = set()

    for doc_id, requirement in requirements.items():
        candidate = document_index.by_id.get(doc_id)
        if candidate is None:
            missing_gold_docs.append(requirement)
            continue
        if doc_id not in selected_doc_ids:
            selected_doc_ids.add(doc_id)
            ordered_doc_ids.append(doc_id)
            found_gold_doc_ids.add(doc_id)
            gold_source_types.append(candidate.source_type)

    negative_needed = max(0, target_docs - len(ordered_doc_ids))
    same_sources = list(dict.fromkeys(gold_source_types + [
        source_type
        for question in selected_questions
        for source_type in question.source_types
        if source_type in source_types
    ]))
    same_source_target = int(round(negative_needed * 0.7))

    same_source_ids = draw_from_sources(
        same_sources,
        count=same_source_target,
        document_index=document_index,
        selected_doc_ids=selected_doc_ids,
        rng=rng,
    )
    ordered_doc_ids.extend(same_source_ids)

    remaining_needed = max(0, target_docs - len(ordered_doc_ids))
    cross_sources = [source_type for source_type in source_types if source_type not in same_sources]
    cross_source_ids = draw_from_sources(
        cross_sources,
        count=remaining_needed,
        document_index=document_index,
        selected_doc_ids=selected_doc_ids,
        rng=rng,
    )
    ordered_doc_ids.extend(cross_source_ids)

    remaining_needed = max(0, target_docs - len(ordered_doc_ids))
    if remaining_needed:
        filler_ids = draw_from_sources(
            source_types,
            count=remaining_needed,
            document_index=document_index,
            selected_doc_ids=selected_doc_ids,
            rng=rng,
        )
        ordered_doc_ids.extend(filler_ids)

    negative_doc_ids = [doc_id for doc_id in ordered_doc_ids if doc_id not in found_gold_doc_ids]
    same_source_negative_count = sum(
        1
        for doc_id in negative_doc_ids
        if document_index.by_id[doc_id].source_type in same_sources
    )
    cross_source_negative_count = len(negative_doc_ids) - same_source_negative_count
    stats = {
        "gold_doc_count": len(found_gold_doc_ids),
        "missing_gold_doc_count": len(missing_gold_docs),
        "same_source_negative_doc_count": same_source_negative_count,
        "cross_source_negative_doc_count": cross_source_negative_count,
        "negative_doc_count": len(negative_doc_ids),
    }
    return ordered_doc_ids, missing_gold_docs, stats


def prepare_output_root(output_root: Path) -> tuple[Path, Path]:
    questions_path = output_root / "questions_mini.jsonl"
    documents_output_root = output_root / "documents"
    if documents_output_root.exists():
        first_existing_file = next(
            (path for path in documents_output_root.rglob("*") if path.is_file()),
            None,
        )
        if first_existing_file is not None:
            raise ValueError(
                "Refusing to write into a non-empty documents directory: "
                f"{documents_output_root}. Choose a new --output-root or remove old files manually."
            )
    output_root.mkdir(parents=True, exist_ok=True)
    documents_output_root.mkdir(parents=True, exist_ok=True)
    return questions_path, documents_output_root


def output_path_text(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_questions(path: Path, selected_questions: list[QuestionCandidate]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in selected_questions:
            handle.write(json.dumps(candidate.record, ensure_ascii=False) + "\n")


def format_json_document(candidate: DocumentCandidate) -> str:
    title = candidate.title or ""
    content = candidate.content or ""
    return "\n".join(
        [
            f"Title: {title}",
            "",
            f"Source Type: {candidate.source_type}",
            f"Document ID: {candidate.doc_id}",
            "",
            content,
            "",
        ]
    )


def write_document(candidate: DocumentCandidate, documents_output_root: Path) -> Path:
    source_dir = documents_output_root / candidate.source_type
    source_dir.mkdir(parents=True, exist_ok=True)
    output_path = source_dir / f"{candidate.doc_id}.txt"

    if candidate.record is not None:
        output_path.write_text(format_json_document(candidate), encoding="utf-8")
        return output_path

    if candidate.input_path.suffix.lower() == ".txt":
        shutil.copyfile(candidate.input_path, output_path)
    else:
        text = candidate.input_path.read_text(encoding="utf-8", errors="replace")
        output_path.write_text(text, encoding="utf-8")
    return output_path


def write_documents(
    doc_ids: list[str],
    document_index: DocumentIndex,
    documents_output_root: Path,
    output_root: Path,
) -> dict[str, str]:
    doc_id_to_path: dict[str, str] = {}
    for doc_id in doc_ids:
        candidate = document_index.by_id[doc_id]
        output_path = write_document(candidate, documents_output_root)
        doc_id_to_path[doc_id] = str(output_path.relative_to(output_root)).replace("\\", "/")
    return doc_id_to_path


def build_manifest(
    *,
    args: argparse.Namespace,
    questions_path: Path,
    documents_root: Path,
    output_root: Path,
    output_questions_file: Path,
    output_documents_root: Path,
    selected_questions: list[QuestionCandidate],
    selected_doc_ids: list[str],
    missing_gold_docs: list[dict[str, Any]],
    doc_stats: dict[str, int],
    document_index: DocumentIndex,
) -> dict[str, Any]:
    question_type_counts = Counter(
        str(candidate.record.get("question_type") or "") for candidate in selected_questions
    )
    source_type_counts = Counter(
        source_type
        for candidate in selected_questions
        for source_type in candidate.source_types
    )
    document_source_type_counts = Counter(
        document_index.by_id[doc_id].source_type for doc_id in selected_doc_ids
    )

    warnings: list[str] = []
    if len(selected_questions) < args.max_questions:
        warnings.append(
            f"Only selected {len(selected_questions)} questions because eligible questions were insufficient."
        )
    if len(selected_doc_ids) < args.target_docs:
        warnings.append(
            f"Only selected {len(selected_doc_ids)} documents because available documents were insufficient."
        )
    if missing_gold_docs:
        warnings.append(f"{len(missing_gold_docs)} gold documents were not found.")
    if document_index.duplicate_doc_id_count:
        warnings.append(
            f"{document_index.duplicate_doc_id_count} duplicate document IDs were found; kept first path."
        )

    return {
        "seed": args.seed,
        "max_questions": args.max_questions,
        "target_docs": args.target_docs,
        "actual_questions": len(selected_questions),
        "actual_docs": len(selected_doc_ids),
        "question_type_counts": dict(question_type_counts),
        "source_type_counts": dict(source_type_counts),
        "document_source_type_counts": dict(document_source_type_counts),
        "gold_doc_count": doc_stats["gold_doc_count"],
        "missing_gold_doc_count": len(missing_gold_docs),
        "negative_doc_count": doc_stats["negative_doc_count"],
        "same_source_negative_doc_count": doc_stats["same_source_negative_doc_count"],
        "cross_source_negative_doc_count": doc_stats["cross_source_negative_doc_count"],
        "duplicate_doc_ids": document_index.duplicate_doc_ids,
        "duplicate_doc_id_count": document_index.duplicate_doc_id_count,
        "questions_file": str(questions_path),
        "documents_root": str(documents_root),
        "output_questions_file": output_path_text(output_questions_file),
        "output_documents_root": output_path_text(output_documents_root),
        "scanned_document_files": document_index.scanned_files,
        "scanned_document_records": document_index.scanned_records,
        "skipped_document_files": document_index.skipped_files,
        "skipped_document_records": document_index.skipped_records,
        "warnings": warnings,
    }


def print_next_commands(output_root: Path) -> None:
    questions_file = output_root / "questions_mini.jsonl"
    documents_root = output_root / "documents"
    print("")
    print("[INFO] Next commands:")
    print(
        "python scripts/ingest.py "
        f"--path {output_path_text(documents_root)} "
        "--collection enterprise_mini"
    )
    print(
        "python scripts/run_enterprise_rag_eval.py "
        f"--questions-file {output_path_text(questions_file)} "
        "--collection enterprise_mini "
        "--modes dense bm25 hybrid hybrid_rerank "
        "--top-k 10 "
        "--candidate-k 30 "
        "--markdown"
    )


def build_subset(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    questions_path = resolve_questions_file(args.questions_file)
    documents_root = resolve_documents_root(args.documents_root)
    output_root = resolve_project_path(args.output_root)
    source_types = {normalise_label(item) for item in args.source_types if str(item).strip()}

    log_info(f"Questions file: {questions_path}")
    log_info(f"Documents root: {documents_root}")
    log_info(f"Output root: {output_root}")

    selected_questions = select_questions(
        questions_path,
        max_questions=args.max_questions,
        requested_question_types=args.question_types,
        source_types=source_types,
        rng=rng,
    )
    log_info(f"Selected questions: {len(selected_questions)}")

    document_index = build_document_index(documents_root, source_types=source_types)
    log_info(
        "Indexed documents: "
        f"{len(document_index.by_id)} unique IDs from {document_index.scanned_files} files"
    )
    if document_index.duplicate_doc_id_count:
        log_warn(f"Duplicate document IDs found: {document_index.duplicate_doc_id_count}")

    selected_doc_ids, missing_gold_docs, doc_stats = select_documents(
        selected_questions,
        document_index,
        target_docs=args.target_docs,
        source_types=source_types,
        rng=rng,
    )
    log_info(
        "Selected documents: "
        f"{len(selected_doc_ids)} total, {doc_stats['gold_doc_count']} gold found, "
        f"{len(missing_gold_docs)} gold missing"
    )

    output_questions_file, output_documents_root = prepare_output_root(output_root)
    write_questions(output_questions_file, selected_questions)
    doc_id_to_path = write_documents(
        selected_doc_ids,
        document_index,
        output_documents_root,
        output_root,
    )

    (output_root / "doc_id_to_path.json").write_text(
        json.dumps(doc_id_to_path, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "missing_gold_docs.json").write_text(
        json.dumps(missing_gold_docs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = build_manifest(
        args=args,
        questions_path=questions_path,
        documents_root=documents_root,
        output_root=output_root,
        output_questions_file=output_questions_file,
        output_documents_root=output_documents_root,
        selected_questions=selected_questions,
        selected_doc_ids=selected_doc_ids,
        missing_gold_docs=missing_gold_docs,
        doc_stats=doc_stats,
        document_index=document_index,
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log_info(f"Wrote questions: {output_questions_file}")
    log_info(f"Wrote documents: {output_documents_root}")
    log_info(f"Wrote manifest: {output_root / 'manifest.json'}")
    if missing_gold_docs:
        log_warn(f"Missing gold document details: {output_root / 'missing_gold_docs.json'}")
    print_next_commands(output_root)
    return manifest


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)
    try:
        build_subset(args)
    except Exception as exc:
        print(f"[FAIL] Failed to build EnterpriseRAG-Bench mini subset: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
