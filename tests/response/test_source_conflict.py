from src.core.response.source_conflict import SourceConflictDetector


def test_conflicting_statuses_generate_warning() -> None:
    contexts = [
        type("Context", (), {"citation_id": "C1", "text": "功能已启用。"})(),
        type("Context", (), {"citation_id": "C2", "text": "功能未启用。"})(),
    ]

    result = SourceConflictDetector().detect(contexts)

    assert result.detected is True
    assert "CONFLICTING_SOURCES" in result.warnings

