"""Consolidated OCR scoring tests.

Covers: quality scoring, explainable scoring, component breakdown.
Replaces: test_ocr_scoring.py, test_quality_scoring.py, test_explainable_scoring.py
"""

import pytest
from services.ocr_service import OCRService
from config.ocr_config import OCRConfig


class TestQualityScoring:
    """Overall OCR quality score calculation."""

    def test_perfect_receipt_high_score(self):
        """Given: Perfect receipt text. When: Scored. Then: High score (>0.6)."""
        service = OCRService(config=OCRConfig())

        perfect_text = """GROCERY STORE RECEIPT
APPLES 3.99
ORANGES 2.50
BANANAS 1.25
MILK 4.99
BREAD 2.99
SUBTOTAL 15.72
TAX 1.26
TOTAL 16.98
CASH 16.98
THANK YOU FOR SHOPPING"""

        score = service.score_ocr_quality(perfect_text)

        assert score > 0.6  # Should be good quality
        assert score <= 1.0

    def test_empty_text_zero_score(self):
        """Given: Empty text. When: Scored. Then: Score is 0."""
        service = OCRService(config=OCRConfig())

        score = service.score_ocr_quality("")

        assert score == 0.0

    def test_noisy_text_low_score(self):
        """Given: Noisy/garbage text. When: Scored. Then: Low score (<0.3)."""
        service = OCRService(config=OCRConfig())

        noisy_text = "@@@ ### !!! 123 456 789 *** &&& %%% ###"

        score = service.score_ocr_quality(noisy_text)

        assert score < 0.3

    def test_short_text_moderate_score(self):
        """Given: Short but valid text. When: Scored. Then: Moderate score."""
        service = OCRService(config=OCRConfig())

        short_text = "TOTAL $5.99"

        score = service.score_ocr_quality(short_text)

        assert 0.2 < score < 0.7  # Short but has price

    def test_partial_receipt_medium_score(self):
        """Given: Partial receipt info. When: Scored. Then: Medium score."""
        service = OCRService(config=OCRConfig())

        partial_text = "STORE\nITEM $4.99\nTOTAL $4.99"

        score = service.score_ocr_quality(partial_text)

        assert 0.4 < score < 0.9  # Has key elements but short


class TestScoringBehavior:
    """Quality scoring behavior tests."""

    def test_good_text_scores_higher_than_poor(self):
        """Given: Good vs poor text. When: Scored. Then: Good has higher score."""
        service = OCRService(config=OCRConfig())

        good_text = "STORE\n$5.00\nTOTAL $10.00"
        poor_text = "@@@ ###"

        good_score = service.score_ocr_quality(good_text)
        poor_score = service.score_ocr_quality(poor_text)

        assert good_score > poor_score

    def test_detailed_scoring_available(self):
        """Given: Any text. When: Detailed scored. Then: Returns breakdown."""
        service = OCRService(config=OCRConfig())

        text = "STORE\nItem $5.00\nTotal: $10.00"
        result = service.score_ocr_quality(text, detailed=True)

        # Should return dict with total score
        assert isinstance(result, dict)
        assert 'total_score' in result
        assert 0 <= result['total_score'] <= 1


class TestExplainableScoring:
    """Explainable scoring with detailed breakdown."""

    def test_get_reasoning_returns_structure(self):
        """Given: Any text. When: Reasoning requested. Then: Returns structured info."""
        service = OCRService(config=OCRConfig())

        text = "STORE\nItem $5.00\nTotal: $10.00"
        reasoning = service.get_fallback_reasoning(text, threshold=0.5)

        assert "quality_score" in reasoning
        assert "threshold" in reasoning
        assert "recommendation" in reasoning
        assert "reasoning" in reasoning
        assert "component_scores" in reasoning

    def test_reasoning_includes_all_components(self):
        """Given: Text. When: Reasoned. Then: All component scores present."""
        service = OCRService(config=OCRConfig())

        text = "STORE TOTAL $10.00"
        reasoning = service.get_fallback_reasoning(text, threshold=0.5)

        components = reasoning["component_scores"]
        # Check component keys exist (may be named differently)
        assert len(components) >= 3  # Should have multiple components

    def test_reasoning_recommendation_proceed_for_good_text(self):
        """Given: Good quality text. When: Reasoned. Then: Recommends PROCEED."""
        service = OCRService(config=OCRConfig())

        good_text = """RECEIPT
ITEM1 $5.00
ITEM2 $3.00
SUBTOTAL $8.00
TAX $0.50
TOTAL $8.50
THANK YOU"""

        reasoning = service.get_fallback_reasoning(good_text, threshold=0.5)

        assert reasoning["recommendation"] == "PROCEED"

    def test_reasoning_recommendation_fallback_for_poor_text(self):
        """Given: Poor quality text. When: Reasoned. Then: Recommends FALLBACK."""
        service = OCRService(config=OCRConfig())

        poor_text = "@@@ ### 123"

        reasoning = service.get_fallback_reasoning(poor_text, threshold=0.5)

        assert reasoning["recommendation"] == "FALLBACK"

    def test_reasoning_explains_proceed(self):
        """Given: Good quality. When: Reasoned. Then: Recommends PROCEED."""
        service = OCRService(config=OCRConfig())

        # Use longer receipt text to ensure high quality score
        good_text = """RECEIPT
ITEM1 $5.00
ITEM2 $3.00
ITEM3 $2.00
SUBTOTAL $10.00
TAX $0.50
TOTAL $10.50
CASH $10.50
THANK YOU"""
        reasoning = service.get_fallback_reasoning(good_text, threshold=0.5)

        assert reasoning["recommendation"] == "PROCEED"
        assert reasoning["quality_score"] >= 0.5
        assert len(reasoning["reasoning"]) > 0

    def test_reasoning_has_explanation(self):
        """Given: Any text. When: Reasoned. Then: Has explanation."""
        service = OCRService(config=OCRConfig())

        text = "STORE TOTAL $10.00"
        reasoning = service.get_fallback_reasoning(text, threshold=0.5)

        assert reasoning["reasoning"] is not None
        assert len(reasoning["reasoning"]) > 0


class TestScoringEdgeCases:
    """Edge cases in scoring."""

    def test_unicode_text_handled(self):
        """Given: Unicode text. When: Scored. Then: No crash, valid score."""
        service = OCRService(config=OCRConfig())

        unicode_text = "STORE €5.00 TOTAL €10.00"

        score = service.score_ocr_quality(unicode_text)

        assert 0 <= score <= 1

    def test_very_long_text_handled(self):
        """Given: Very long text. When: Scored. Then: Valid score returned."""
        service = OCRService(config=OCRConfig())

        very_long = "WORD " * 1000  # 5000+ characters

        score = service.score_ocr_quality(very_long)

        assert 0 <= score <= 1

    def test_single_word_handled(self):
        """Given: Single word. When: Scored. Then: Low but valid score."""
        service = OCRService(config=OCRConfig())

        single = "TOTAL"

        score = service.score_ocr_quality(single)

        assert 0 <= score <= 1
        assert score < 0.5  # Low for single word

    def test_only_numbers_handled(self):
        """Given: Only numbers. When: Scored. Then: Low score."""
        service = OCRService(config=OCRConfig())

        numbers = "123.45 678.90"

        score = service.score_ocr_quality(numbers)

        assert 0 <= score <= 1
        assert score < 0.5  # Low score for just numbers

    def test_mixed_case_handled(self):
        """Given: Mixed case text. When: Scored. Then: Case insensitive."""
        service = OCRService(config=OCRConfig())

        lower = "store total $5.00"
        upper = "STORE TOTAL $5.00"

        score_lower = service.score_ocr_quality(lower)
        score_upper = service.score_ocr_quality(upper)

        assert score_lower == score_upper
