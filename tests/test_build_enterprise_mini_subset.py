from __future__ import annotations

import json
from pathlib import Path

from scripts.build_enterprise_mini_subset import extract_doc_id, main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_extract_doc_id_from_filename_path_body_and_json() -> None:
    assert (
        extract_doc_id("dsid_ABC123ef__release-notes.txt")
        == "dsid_abc123ef"
    )
    assert (
        extract_doc_id(r"C:\docs\github\dsid_deadbeef.txt")
        == "dsid_deadbeef"
    )
    assert extract_doc_id("Document ID: dsid_feedface") == "dsid_feedface"
    assert extract_doc_id({"doc_id": "dsid_0123abcd"}) == "dsid_0123abcd"


def test_build_subset_outputs_questions_docs_and_manifest(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    docs_root = tmp_path / "all_documents"
    output_root = tmp_path / "mini"

    write_jsonl(
        questions_path,
        [
            {
                "question_id": "q1",
                "question_type": "basic",
                "source_types": ["github"],
                "question": "What changed?",
                "expected_doc_ids": ["dsid_goldgithub"],
                "gold_answer": "A",
                "answer_facts": ["A"],
            },
            {
                "question_id": "q2",
                "question_type": "semantic",
                "source_types": ["slack"],
                "question": "Who approved it?",
                "expected_doc_ids": ["dsid_goldslack"],
            },
            {
                "question_id": "q3",
                "question_type": "constrained",
                "source_types": ["jira"],
                "question": "Which ticket?",
                "expected_doc_ids": ["dsid_missing"],
            },
            {
                "question_id": "q4",
                "question_type": "intra_document_reasoning",
                "source_types": ["confluence"],
                "question": "What policy applies?",
                "expected_doc_ids": ["dsid_goldconf"],
            },
        ],
    )

    for source_type, doc_id in [
        ("github", "dsid_goldgithub"),
        ("slack", "dsid_goldslack"),
        ("confluence", "dsid_goldconf"),
        ("github", "dsid_neggithub"),
        ("slack", "dsid_negslack"),
        ("gmail", "dsid_neggmail"),
    ]:
        source_dir = docs_root / source_type
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / f"{doc_id}__sample.txt").write_text(
            f"Document ID: {doc_id}\nContent for {doc_id}",
            encoding="utf-8",
        )

    exit_code = main(
        [
            "--questions-file",
            str(questions_path),
            "--documents-root",
            str(docs_root),
            "--output-root",
            str(output_root),
            "--max-questions",
            "4",
            "--target-docs",
            "5",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    assert (output_root / "questions_mini.jsonl").exists()
    assert (output_root / "documents" / "github" / "dsid_goldgithub.txt").exists()
    assert (output_root / "documents" / "slack" / "dsid_goldslack.txt").exists()

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["actual_questions"] == 4
    assert manifest["actual_docs"] == 5
    assert manifest["gold_doc_count"] == 3
    assert manifest["missing_gold_doc_count"] == 1

    missing = json.loads((output_root / "missing_gold_docs.json").read_text(encoding="utf-8"))
    assert missing[0]["doc_id"] == "dsid_missing"


def test_build_subset_exports_jsonl_documents(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    docs_root = tmp_path / "generated_data" / "sources"
    output_root = tmp_path / "mini_json"
    github_dir = docs_root / "github"
    github_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        questions_path,
        [
            {
                "question_id": "q1",
                "question_type": "basic",
                "source_types": ["github"],
                "question": "What is in the JSON document?",
                "expected_doc_ids": ["dsid_jsondoc"],
            }
        ],
    )
    write_jsonl(
        github_dir / "records.jsonl",
        [
            {
                "doc_id": "dsid_jsondoc",
                "source_type": "github",
                "title": "JSON Title",
                "content": "JSON body",
            },
            {
                "doc_id": "dsid_jsonneg",
                "source_type": "github",
                "title": "Negative",
                "content": "Negative body",
            },
        ],
    )

    exit_code = main(
        [
            "--questions-file",
            str(questions_path),
            "--documents-root",
            str(docs_root),
            "--output-root",
            str(output_root),
            "--max-questions",
            "1",
            "--target-docs",
            "2",
            "--source-types",
            "github",
        ]
    )

    assert exit_code == 0
    exported = output_root / "documents" / "github" / "dsid_jsondoc.txt"
    assert "Title: JSON Title" in exported.read_text(encoding="utf-8")
