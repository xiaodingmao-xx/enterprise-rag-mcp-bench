"""Tests for EnterpriseRAG-Bench retrieval evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import scripts.run_enterprise_rag_eval as enterprise_eval
from scripts.run_enterprise_rag_eval import (
    EnterpriseRAGTestCase,
    RagasEvalCache,
    dedupe_keep_order,
    extract_dsid,
    format_markdown_table,
    load_enterprise_questions,
    result_to_doc_id,
    _evaluate_mode,
    _result_texts,
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


def test_result_texts_respects_total_character_budget() -> None:
    results = [
        SimpleNamespace(text="alpha beta"),
        SimpleNamespace(text="gamma delta"),
    ]

    assert _result_texts(results, max_chars=12) == ["alpha beta", "ga"]


def test_evaluate_mode_records_ragas_faithfulness(monkeypatch) -> None:
    retrieved = [
        SimpleNamespace(
            chunk_id="dsid_abcd1234_chunk_0001",
            score=1.0,
            text="The rollout was approved by the platform team.",
            metadata={"source_path": "data/dsid_abcd1234.txt"},
        )
    ]

    def fake_run_retrieval(**kwargs):
        return retrieved

    def fake_serialise_result(result):
        return {
            "chunk_id": result.chunk_id,
            "score": result.score,
            "metadata": result.metadata,
            "text_preview": result.text,
        }

    def fake_ablation_helpers():
        return enterprise_eval.MODES, None, fake_run_retrieval, fake_serialise_result

    class FakeAnswerGenerator:
        def chat(self, messages, **kwargs):  # noqa: ANN001, ANN002
            return SimpleNamespace(
                content="The rollout was approved by the platform team."
            )

    class FakeRagasEvaluator:
        def evaluate(self, **kwargs):  # noqa: ANN003
            return {"faithfulness": 0.75}

    monkeypatch.setattr(enterprise_eval, "_ablation_helpers", fake_ablation_helpers)

    result = _evaluate_mode(
        mode="hybrid",
        test_cases=[
            EnterpriseRAGTestCase(
                question_id="q1",
                query="Who approved the rollout?",
                question_type="basic",
                source_types=["github"],
                expected_doc_ids=["dsid_abcd1234"],
            )
        ],
        top_k=10,
        candidate_k=30,
        components={},
        enable_generation=True,
        enable_ragas=True,
        answer_generator=FakeAnswerGenerator(),
        ragas_evaluator=FakeRagasEvaluator(),
        max_context_chars=1000,
    )

    query_result = result["query_results"][0]
    assert query_result["generated_answer"]
    assert query_result["ragas"] == {"faithfulness": 0.75}
    assert query_result["metrics"]["faithfulness"] == 0.75
    assert result["aggregate_metrics"]["faithfulness"] == 0.75


def test_evaluate_mode_uses_ragas_cache(monkeypatch, tmp_path: Path) -> None:
    retrieved = [
        SimpleNamespace(
            chunk_id="dsid_abcd1234_chunk_0001",
            score=1.0,
            text="The rollout was approved by the platform team.",
            metadata={"source_path": "data/dsid_abcd1234.txt"},
        )
    ]

    def fake_run_retrieval(**kwargs):
        return retrieved

    def fake_serialise_result(result):
        return {
            "chunk_id": result.chunk_id,
            "score": result.score,
            "metadata": result.metadata,
            "text_preview": result.text,
        }

    def fake_ablation_helpers():
        return enterprise_eval.MODES, None, fake_run_retrieval, fake_serialise_result

    calls = {"generation": 0, "ragas": 0}

    class FakeAnswerGenerator:
        def chat(self, messages, **kwargs):  # noqa: ANN001, ANN002
            calls["generation"] += 1
            return SimpleNamespace(
                content="The rollout was approved by the platform team."
            )

    class FakeRagasEvaluator:
        def evaluate(self, **kwargs):  # noqa: ANN003
            calls["ragas"] += 1
            return {"faithfulness": 0.75}

    monkeypatch.setattr(enterprise_eval, "_ablation_helpers", fake_ablation_helpers)

    test_case = EnterpriseRAGTestCase(
        question_id="q1",
        query="Who approved the rollout?",
        question_type="basic",
        source_types=["github"],
        expected_doc_ids=["dsid_abcd1234"],
    )
    cache = RagasEvalCache(tmp_path / "ragas_cache.json")

    first = _evaluate_mode(
        mode="hybrid",
        test_cases=[test_case],
        top_k=10,
        candidate_k=30,
        components={},
        enable_generation=True,
        enable_ragas=True,
        answer_generator=FakeAnswerGenerator(),
        ragas_evaluator=FakeRagasEvaluator(),
        max_context_chars=1000,
        ragas_cache=cache,
        ragas_metrics=["faithfulness"],
    )
    second = _evaluate_mode(
        mode="hybrid",
        test_cases=[test_case],
        top_k=10,
        candidate_k=30,
        components={},
        enable_generation=True,
        enable_ragas=True,
        answer_generator=FakeAnswerGenerator(),
        ragas_evaluator=FakeRagasEvaluator(),
        max_context_chars=1000,
        ragas_cache=cache,
        ragas_metrics=["faithfulness"],
    )

    assert calls == {"generation": 1, "ragas": 1}
    assert first["query_results"][0]["ragas_cache_hit"] is False
    assert second["query_results"][0]["ragas_cache_hit"] is True
    assert second["ragas_cache_hit_count"] == 1
    assert second["aggregate_metrics"]["faithfulness"] == 0.75


def test_markdown_table_includes_ragas_columns() -> None:
    report = {
        "enable_generation": True,
        "enable_ragas": True,
        "ragas_metrics": ["faithfulness"],
        "modes": ["hybrid"],
        "results": {
            "hybrid": {
                "evaluated_query_count": 1,
                "aggregate_metrics": {
                    "recall@10": 1.0,
                    "precision@10": 0.1,
                    "mrr@10": 1.0,
                    "ndcg@10": 1.0,
                    "hit@10": 1.0,
                    "latency_ms": 20.0,
                    "generation_latency_ms": 30.0,
                    "faithfulness": 0.75,
                    "ragas_latency_ms": 40.0,
                },
            }
        },
    }

    markdown = format_markdown_table(report, top_k=10)

    assert "faithfulness" in markdown
    assert "Gen Latency ms" in markdown
    assert "RAGAS Latency ms" in markdown


def test_markdown_table_includes_ragas_cache_hits() -> None:
    report = {
        "enable_generation": True,
        "enable_ragas": True,
        "ragas_cache": "eval/cache/enterprise_rag_ragas_cache.json",
        "ragas_metrics": ["faithfulness"],
        "modes": ["hybrid"],
        "results": {
            "hybrid": {
                "evaluated_query_count": 1,
                "ragas_cache_hit_count": 1,
                "aggregate_metrics": {
                    "recall@10": 1.0,
                    "precision@10": 0.1,
                    "mrr@10": 1.0,
                    "ndcg@10": 1.0,
                    "hit@10": 1.0,
                    "latency_ms": 20.0,
                    "generation_latency_ms": 0.0,
                    "faithfulness": 0.75,
                    "ragas_latency_ms": 0.0,
                },
            }
        },
    }

    markdown = format_markdown_table(report, top_k=10)

    assert "RAGAS Cache Hits" in markdown
    assert "| hybrid | 1 | 1.0000 | 0.1000 | 1.0000 | 1.0000 | 1.0000 | 20.0 | 0.0 | 0.7500 | 0.0 | 1 |" in markdown
