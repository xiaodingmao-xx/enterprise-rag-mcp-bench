from src.core.response.claim_extractor import RuleBasedClaimExtractor


def test_rule_extractor_collects_markers_and_factual_sentences() -> None:
    claims = RuleBasedClaimExtractor().extract("功能已启用，版本为 2.0。[C1] 普通寒暄。")

    assert len(claims) == 1
    assert claims[0].citation_ids == ["C1"]
    assert "2.0" in claims[0].text

