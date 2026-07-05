"""Tests for EnterpriseRAG-Bench retrieval evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_enterprise_rag_eval import (
    dedupe_keep_order,
    extract_dsid,
    load_enterprise_questions,
    result_to_doc_id,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_extract_dsid_from_path() -> None:
    value = r"G:\data\docs\github\dsid_0A1b2C3d.txt"

    assert extract_dsid(value) == "dsid_0a1b2c3d"


def test_dedupe_keep_order_skips_empty_values() -> None:
    items = ["dsid_a", "", "dsid_b", "dsid_a", "  ", "dsid_c"]

    assert dedupe_keep_order(items) == ["dsid_a", "dsid_b", "dsid_c"]


def test_load_enterprise_questions_filters_jsonl(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    _write_jsonl(
        questions_path,
        [
            {
                "question_id": "qst_0001",
                "question_type": "basic",
                "source_types": ["github"],
                "question": "What changed in the API?",
                "expected_doc_ids": ["dsid_aaa111"],
                "gold_answer": "The API changed.",
                "answer_facts": ["fact 1"],
            },
            {
                "question_id": "qst_0002",
                "question_type": "semantic",
                "source_types": ["gmail"],
                "question": "Who approved the rollout?",
                "expected_doc_ids": ["dsid_bbb222"],
            },
            {
                "question_id": "qst_0003",
                "question_type": "basic",
                "source_types": ["github", "linear"],
                "question": "Missing ground truth?",
                "expected_doc_ids": [],
            },
            {
                "question_id": "qst_0004",
                "question_type": "basic",
                "source_types": ["linear"],
                "question": "Which ticket tracks this?",
                "expected_doc_ids": ["dsid_ccc333"],
            },
        ],
    )

    cases = load_enterprise_questions(
        questions_path,
        question_types={"basic"},
        source_types={"github"},
        max_questions=1,
    )

    assert len(cases) == 1
    assert cases[0].question_id == "qst_0001"
    assert cases[0].query == "What changed in the API?"
    assert cases[0].question_type == "basic"
    assert cases[0].source_types == ["github"]
    assert cases[0].expected_doc_ids == ["dsid_aaa111"]
    assert cases[0].gold_answer == "The API changed."
    assert cases[0].answer_facts == ["fact 1"]


def test_load_enterprise_questions_can_include_no_ground_truth(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    _write_jsonl(
        questions_path,
        [
            {
                "question_id": "qst_empty",
                "question_type": "info_not_found",
                "source_types": [],
                "question": "Unknown?",
                "expected_doc_ids": [],
            }
        ],
    )

    assert load_enterprise_questions(questions_path) == []

    cases = load_enterprise_questions(
        questions_path,
        skip_no_ground_truth=False,
    )

    assert len(cases) == 1
    assert cases[0].expected_doc_ids == []


def test_result_to_doc_id_extracts_from_dict_metadata() -> None:
    result = {
        "chunk_id": "chunk_without_doc_id",
        "metadata": {
            "source_path": r"C:\enterprise\github\dsid_abcd1234.md",
        },
    }

    assert result_to_doc_id(result) == "dsid_abcd1234"


def test_result_to_doc_id_falls_back_to_chunk_id() -> None:
    result = {
        "chunk_id": "dsid_deadbeef_chunk_0001",
        "metadata": {},
    }

    assert result_to_doc_id(result) == "dsid_deadbeef"
