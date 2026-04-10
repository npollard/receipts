"""Pipeline behavior tests using harness + fakes.

Refactored from mock-based tests to scenario-based behavior validation.
These tests verify what the pipeline does, not how it's implemented.
"""

import pytest
from pathlib import Path
from tests.harness.pipeline_harness import PipelineTestHarness, PipelineStatus
from tests.harness.builders import ReceiptBuilder
from tests.harness.fakes import OCROutput
from tests.harness.fakes.fake_receipt_parser import ParserOutput
from core.exceptions import OCRError


class TestBasicPipelineExecution:
    """Core pipeline execution scenarios."""

    def test_successful_receipt_processing(self):
        """Given: Valid receipt image. When: Processed. Then: Success with data."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "GROCERY $25.50", quality=0.90)
        harness.parser.set_parse_result(merchant="Grocery Store", total=25.50)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_data["merchant_name"] == "Grocery Store"
        assert result.receipt_data["total_amount"] == 25.50
        harness.assert_persisted_once()
        harness.assert_trace_stage_sequence(["OCR", "PARSE", "VALIDATE", "PERSIST"])

    def test_pipeline_with_user_context(self):
        """Given: User context. When: Receipt processed. Then: Associated with user."""
        harness = PipelineTestHarness()
        harness.with_user("user_123")

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=10.00)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        records = harness.get_persisted_records()
        assert len(records) == 1

    def test_nonexistent_image_fails_gracefully(self):
        """Given: Non-existent image. When: Processed. Then: Graceful failure."""
        harness = PipelineTestHarness()

        # OCR configured to fail (simulating missing file)
        harness.ocr.set_should_fail(OCRError("File not found"))

        result = harness.run("nonexistent.jpg")

        assert result.status == PipelineStatus.FAILED
        assert "OCR" in result.error_message or result.error_message is not None
        harness.assert_not_persisted()


class TestPipelineRetryBehavior:
    """Pipeline retry and recovery scenarios."""

    def test_ocr_retry_eventually_succeeds(self):
        """Given: OCR fails twice then succeeds. When: Processed. Then: Success."""
        harness = PipelineTestHarness()

        harness.ocr.set_sequence([
            OCRError("Network timeout"),
            OCRError("Service busy"),
            OCROutput(text="STORE $30.00", quality_score=0.88),
        ])

        harness.parser.set_parse_result(merchant="Store", total=30.00)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        harness.assert_trace_stage_count("OCR", 3)
        harness.assert_trace_retry_sequence("OCR", 3)

    def test_parser_retry_with_fake_retry_service(self):
        """Given: Parser fails once. When: Retried. Then: Success on second attempt."""
        harness = PipelineTestHarness(use_fake_retry=True)

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $40.00", quality=0.85)
        harness.parser.set_sequence([
            ValueError("Parse error"),
            ParserOutput(merchant="Store", total=40.00),
        ])
        harness.retry.set_succeed_on_attempt(2)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        assert result.retry_count == 1
        harness.assert_persisted_once()


class TestPipelineFallbackBehavior:
    """OCR fallback scenarios."""

    def test_low_quality_ocr_triggers_fallback(self):
        """Given: Low quality EasyOCR. When: Processed. Then: Vision fallback used."""
        harness = PipelineTestHarness()

        harness.ocr.set_sequence([
            OCROutput(text="BLURRY", quality_score=0.15, method="easyocr"),
            OCROutput(text="CLEAR STORE $50.00", quality_score=0.90, method="vision"),
        ])

        harness.parser.set_parse_result(merchant="Store", total=50.00)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        assert result.ocr_method == "vision"
        harness.assert_trace_stage_count("OCR", 2)
        harness.assert_ocr_used_fallback()

    def test_fallback_failure_aborts_pipeline(self):
        """Given: Both OCR methods fail. When: Processed. Then: Pipeline fails."""
        harness = PipelineTestHarness()

        harness.ocr.set_sequence([
            OCRError("EasyOCR failed"),
            OCRError("Vision API failed"),
        ])

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED
        harness.assert_not_persisted()


class TestPipelineValidationBehavior:
    """Validation and partial result scenarios."""

    def test_validation_failure_creates_partial_result(self):
        """Given: Some fields invalid. When: Processed with preserve. Then: Partial saved."""
        from tests.harness.fakes.fake_validation_service import ValidationField

        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $20.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=20.00, date="")

        harness.validator.field_passes(ValidationField.MERCHANT)
        harness.validator.field_passes(ValidationField.TOTAL)
        harness.validator.field_fails(ValidationField.DATE, "Date required")
        harness.validator.set_preserve_partial(True)

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.PARTIAL
        harness.assert_persisted_once()

    def test_critical_validation_failure_aborts(self):
        """Given: All fields invalid. When: Processed without preserve. Then: Fail."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "???", quality=0.85)
        harness.parser.set_parse_result(merchant="", total=None)

        harness.validator.all_fields_fail("Critical error")
        harness.validator.set_preserve_partial(False)

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED
        harness.assert_not_persisted()


class TestPipelineIdempotency:
    """Idempotency and duplicate detection scenarios."""

    def test_duplicate_receipt_detected(self):
        """Given: Same receipt processed twice. When: Second run. Then: Duplicate detected."""
        harness = PipelineTestHarness()

        # Same configuration for both runs
        harness.ocr.set_text_for_image("receipt.jpg", "STORE $35.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=35.00)
        harness.validator.all_fields_pass()

        # First run
        result1 = harness.run("receipt.jpg")
        assert result1.status == PipelineStatus.SUCCESS

        # Second run (same image = duplicate)
        result2 = harness.run("receipt.jpg")
        assert result2.was_duplicate or result2.status == PipelineStatus.DUPLICATE

        # Only one persisted
        harness.assert_persist_count(1)
        harness.assert_unique_hashes(1)

    def test_different_receipts_persisted_separately(self):
        """Given: Two different receipts. When: Processed. Then: Two records."""
        harness = PipelineTestHarness()
        harness.validator.all_fields_pass()

        # First receipt
        harness.ocr.set_text_for_image("r1.jpg", "STORE A $15.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store A", total=15.00)
        result1 = harness.run("r1.jpg")
        assert result1.status == PipelineStatus.SUCCESS

        # Second receipt (different)
        harness.ocr.set_text_for_image("r2.jpg", "STORE B $25.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store B", total=25.00)
        result2 = harness.run("r2.jpg")
        assert result2.status == PipelineStatus.SUCCESS

        # Two distinct records
        harness.assert_persist_count(2)
        harness.assert_unique_hashes(2)


class TestPipelineTraceAndDebugging:
    """Execution trace and debugging scenarios."""

    def test_happy_path_trace(self):
        """Given: Normal flow. When: Processed. Then: Complete trace available."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $45.00", quality=0.90)
        harness.parser.set_parse_result(merchant="Store", total=45.00)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")

        trace = harness.trace.get_all()
        assert len(trace) == 4

        stages = [entry.stage for entry in trace]
        assert stages == ["OCR", "PARSE", "VALIDATE", "PERSIST"]

        # All succeeded
        assert all(entry.success for entry in trace)

    def test_failure_trace_for_debugging(self):
        """Given: Failure scenario. When: Processed. Then: Trace shows failure point."""
        harness = PipelineTestHarness()

        harness.ocr.set_should_fail(OCRError("OCR unavailable"))

        harness.run("receipt.jpg")

        # Trace shows OCR failures
        ocr_entries = [e for e in harness.trace.get_all() if e.stage == "OCR"]
        assert len(ocr_entries) == 3  # Retry attempts
        assert all(not e.success for e in ocr_entries)

        # No downstream stages
        assert harness.trace.get_stage_attempts("PARSE") == 0

    def test_full_report_contains_all_metrics(self):
        """Given: Any scenario. When: Report generated. Then: Complete metrics available."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $55.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=55.00)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")
        report = harness.get_full_report()

        assert "status" in report
        assert "stages" in report
        assert "repository_metrics" in report
        assert "persisted_count" in report
        assert "unique_hashes" in report


class TestBatchScenarios:
    """Batch processing scenarios using scenarios builder."""

    def test_happy_path_scenario(self):
        """Given: Happy path scenario. When: Executed. Then: Success."""
        from tests.harness.builders.scenarios import make_happy_path

        harness = make_happy_path(merchant="Test Store", total=99.99)
        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        harness.assert_persisted_once()

    def test_fallback_scenario(self):
        """Given: Fallback scenario. When: Executed. Then: Uses Vision, succeeds."""
        from tests.harness.builders.scenarios import make_easyocr_fail_then_vision_success

        harness = make_easyocr_fail_then_vision_success(merchant="Restaurant", total=75.00)
        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        assert result.ocr_method == "vision"
        harness.assert_ocr_used_fallback()

    def test_duplicate_detection_scenario(self):
        """Given: Duplicate scenario. When: Executed. Then: Duplicate detected."""
        from tests.harness.builders.scenarios import make_duplicate_detected

        harness = make_duplicate_detected(merchant="Existing Store", total=100.00)
        result = harness.run("receipt.jpg")

        assert result.was_duplicate
        harness.assert_not_persisted()
