"""Tests using scenario helpers - demonstrating specification-style testing.

These tests show how scenarios make tests read like specifications
with minimal boilerplate and clear intent.
"""

import pytest
from tests.harness.pipeline_harness import PipelineStatus
from tests.harness.builders import (
    make_happy_path,
    make_easyocr_fail_then_vision_success,
    make_validation_failure_then_retry_success,
    make_duplicate_detected,
    make_partial_save_with_validation_errors,
    make_full_failure_ocr_error,
    make_full_failure_validation_critical,
    make_retry_exhausted_all_attempts_fail,
    make_complex_fallback_and_retry,
    ScenarioBuilder,
)


class TestHappyPath:
    """Standard successful processing."""

    def test_happy_path_succeeds(self):
        """Given: Clean receipt. When: Processed. Then: Success."""
        harness = make_happy_path(merchant="Whole Foods", total=47.83)

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_data["merchant_name"] == "Whole Foods"
        assert result.receipt_data["total_amount"] == 47.83
        harness.assert_persisted_once()

    def test_happy_path_uses_easyocr(self):
        """Given: High quality OCR. When: Processed. Then: Uses EasyOCR."""
        harness = make_happy_path(ocr_quality=0.90)

        result = harness.run("receipt.jpg")

        assert result.ocr_method == "easyocr"
        assert not harness.ocr.used_fallback()


class TestOCRFallback:
    """Low quality OCR triggers Vision API fallback."""

    def test_fallback_used_when_quality_low(self):
        """Given: Low quality EasyOCR. When: Processed. Then: Vision fallback used."""
        harness = make_easyocr_fail_then_vision_success(
            merchant="Fancy Restaurant",
            total=89.50
        )

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        harness.assert_ocr_used_fallback()
        assert result.ocr_method == "vision"

    def test_fallback_data_parsed_correctly(self):
        """Given: Vision fallback succeeds. When: Parsed. Then: Data extracted."""
        harness = make_easyocr_fail_then_vision_success(total=123.45)

        result = harness.run("receipt.jpg")

        assert result.receipt_data["total_amount"] == 123.45


class TestRetryBehavior:
    """Parser retries on parse failure."""

    def test_parse_error_triggers_retry(self):
        """Given: Parser fails first attempt. When: Retried. Then: Success."""
        harness = make_validation_failure_then_retry_success(
            merchant="Corner Store",
            total=15.00
        )

        result = harness.run("receipt.jpg")

        # After retry, we have valid data with date
        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_data["receipt_date"] == "2024-01-15"
        harness.assert_retry_count(1)

    def test_retry_tracks_strategy(self):
        """Given: Parser fails. When: Retried. Then: Strategy recorded."""
        harness = make_validation_failure_then_retry_success()

        harness.run("receipt.jpg")

        # Strategy was selected based on error type (ValueError -> LLM_SELF_CORRECTION)
        strategies = harness.retry.get_strategies_used()
        assert len(strategies) > 0
        # Strategy configured for ValueError is used
        assert strategies[0] == "LLM_SELF_CORRECTION"

    def test_all_retries_exhausted_pipeline_fails(self):
        """Given: Parser always fails. When: Retried 3 times. Then: Pipeline fails."""
        harness = make_retry_exhausted_all_attempts_fail(max_retries=3)

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED
        harness.assert_not_persisted()


class TestIdempotency:
    """Duplicate detection prevents re-processing."""

    def test_duplicate_detected_returns_existing(self):
        """Given: Same image processed before. When: Processed again. Then: Returns existing."""
        harness = make_duplicate_detected(
            merchant="Previous Store",
            total=99.00
        )

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.DUPLICATE
        assert result.was_duplicate
        harness.assert_not_persisted()


class TestPartialResults:
    """Partial data preservation on validation errors."""

    def test_partial_save_when_some_fields_invalid(self):
        """Given: Date missing, merchant valid. When: Validated. Then: Partial saved."""
        harness = make_partial_save_with_validation_errors(
            merchant="Partial Store",
            total=33.50
        )

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.PARTIAL
        assert result.receipt_data["merchant_name"] == "Partial Store"
        harness.assert_persisted_once()


class TestFullFailures:
    """Complete pipeline failures."""

    def test_ocr_failure_aborts_pipeline(self):
        """Given: Image unreadable. When: OCR attempted. Then: Pipeline fails."""
        harness = make_full_failure_ocr_error()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED
        harness.assert_not_persisted()
        # 3 OCR retry attempts tracked (normalized to single OCR)
        harness.assert_stage_sequence(["OCR"])  # Stops after OCR retries

    def test_critical_validation_failure_no_save(self):
        """Given: All fields invalid. When: Validated. Then: No persistence."""
        harness = make_full_failure_validation_critical()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED
        harness.assert_not_persisted()


class TestComplexScenarios:
    """Complex multi-stage interactions."""

    def test_fallback_then_retry_succeeds(self):
        """Given: Low OCR quality AND parse fails once.
        When: Fallback then retry.
        Then: Success."""
        harness = make_complex_fallback_and_retry(
            merchant="Complex Store",
            total=67.89
        )

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        assert result.ocr_method == "vision"
        harness.assert_retry_count(1)
        harness.assert_persisted_once()


class TestScenarioBuilder:
    """Using fluent ScenarioBuilder for custom cases."""

    def test_custom_scenario_with_builder(self):
        """Given: Custom configuration via builder. Then: Works as expected."""
        harness = (ScenarioBuilder()
            .with_high_quality_ocr("SHOP $25.00")
            .with_parsed_receipt(merchant="Shop", total=25.00, date="2024-06-01")
            .with_validation_passing()
            .build())

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_data["total_amount"] == 25.00

    def test_custom_failure_scenario(self):
        """Given: OCR fails via builder. Then: Pipeline fails."""
        from core.exceptions import OCRError

        harness = (ScenarioBuilder()
            .with_ocr_failing(OCRError("Corrupt image"))
            .build())

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED


class TestScenarioReports:
    """Scenario execution reports for debugging."""

    def test_happy_path_report(self):
        """Given: Happy path. When: Executed. Then: Report shows success."""
        harness = make_happy_path()

        harness.run("receipt.jpg")
        report = harness.get_full_report()

        assert report["status"] == "success"
        assert report["repository_saves"] == 1
        assert report["retry_count"] == 0
        assert len(report["stages"]) == 4  # OCR, PARSING, VALIDATION, PERSISTENCE

    def test_complex_scenario_report(self):
        """Given: Fallback + retry. When: Executed. Then: Report shows details."""
        harness = make_complex_fallback_and_retry()

        harness.run("receipt.jpg")
        report = harness.get_full_report()

        assert report["status"] == "success"
        assert report["retry_count"] == 1
        # OCR stages (up to 3 attempts)
        ocr_stages = [s for s in report["stages"] if s["name"] == "OCR"]
        assert len(ocr_stages) <= 3
