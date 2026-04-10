"""Example tests demonstrating PipelineTestHarness usage."""

import pytest
from datetime import datetime
from decimal import Decimal

from tests.harness.pipeline_harness import PipelineTestHarness, PipelineStatus
from tests.harness.builders import ReceiptBuilder
from core.exceptions import OCRError, ValidationError as ReceiptValidationError


class TestHappyPath:
    """Tests for successful pipeline execution."""

    def test_complete_pipeline_success(self):
        """End-to-end successful processing."""
        harness = PipelineTestHarness()

        # Configure components
        harness.ocr.set_text_for_image(
            "receipt.jpg",
            "GROCERY STORE\nMilk $3.99\nTotal $3.99",
            quality=0.85
        )
        harness.parser.set_parse_result(
            merchant="Grocery Store",
            total=3.99,
            date="2024-03-15",
            items=[{"description": "Milk", "price": 3.99}]
        )
        harness.validator.all_fields_pass()

        # Execute
        result = harness.run("receipt.jpg")

        # Assert
        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_data["merchant_name"] == "Grocery Store"
        assert result.receipt_data["total_amount"] == 3.99
        harness.assert_persisted_once()
        harness.assert_stage_sequence(["OCR", "PARSING", "VALIDATION", "PERSISTENCE"])

    def test_pipeline_with_preconfigured_user(self):
        """Pipeline with existing user in repository."""
        harness = PipelineTestHarness()
        harness.with_seeded_user("test@example.com")

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.9)
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_id is not None


class TestOCRFallback:
    """Tests for OCR quality-based fallback behavior."""

    def test_low_quality_ocr_triggers_vision_fallback(self):
        """When EasyOCR quality is low, Vision API is used."""
        harness = PipelineTestHarness()

        # Primary OCR returns low quality
        harness.ocr.set_text_for_image("receipt.jpg", "BLURRY...", quality=0.15)
        # Fallback returns good quality
        harness.ocr.set_fallback_output(
            "receipt.jpg",
            "GROCERY STORE\nMilk $3.99\nTotal $3.99",
            quality=0.90
        )

        harness.parser.set_parse_result(merchant="Grocery Store", total=3.99)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        harness.assert_ocr_used_fallback()
        assert result.ocr_method == "vision"

    def test_high_quality_ocr_no_fallback(self):
        """When EasyOCR quality is good, no fallback needed."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image(
            "receipt.jpg",
            "STORE\nItem $5.00\nTotal $5.00",
            quality=0.85
        )
        harness.parser.set_parse_result(merchant="Store", total=5.0)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        assert result.ocr_method == "easyocr"


class TestRetryBehavior:
    """Tests for retry strategies and error recovery."""

    def test_parser_retry_succeeds_on_second_attempt(self):
        """Parser fails once, then succeeds."""
        harness = PipelineTestHarness(use_fake_retry=True)

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.8)

        # First fails, second succeeds
        harness.parser.set_parse_fails_with(ValueError("Missing date field"))
        harness.parser.set_parse_result(merchant="Store", total=10.0, date="2024-01-01")

        # Configure retry to succeed on 2nd attempt
        harness.retry.set_succeed_on_attempt(2)
        harness.retry.set_strategy_for_error(ValueError, "LLM_SELF_CORRECTION")

        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_data["receipt_date"] == "2024-01-01"
        harness.assert_retry_count(1)  # 1 retry = 2 attempts total
        assert harness.retry.used_strategy("LLM_SELF_CORRECTION")

    def test_all_retries_exhausted_fails_pipeline(self):
        """When all retries fail, pipeline fails."""
        harness = PipelineTestHarness(use_fake_retry=True)

        harness.ocr.set_text_for_image("receipt.jpg", "STORE", quality=0.8)

        # Always fails
        harness.parser.set_parse_fails_with(ValueError("Unparseable"))
        harness.retry.set_max_retries(3)
        harness.retry.set_succeed_on_attempt(99)  # Never succeeds

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED
        harness.assert_not_persisted()


class TestIdempotency:
    """Tests for duplicate detection and idempotency."""

    def test_duplicate_receipt_detected(self):
        """Same image hash returns existing receipt without re-processing."""
        harness = PipelineTestHarness()

        # Use hash that harness will compute from filename (hash_receipt for "receipt.jpg")
        expected_hash = "hash_receipt"

        # Seed existing receipt with matching hash
        existing = (ReceiptBuilder()
            .with_image_hash(expected_hash)
            .with_merchant("Original Store")
            .with_total(25.0)
            .build())
        harness.with_seeded_receipt(existing)

        # Configure OCR
        harness.ocr.set_text_for_image("receipt.jpg", "ANY TEXT")

        # Run pipeline
        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.DUPLICATE
        assert result.was_duplicate
        assert result.receipt_id == existing.id
        harness.assert_not_persisted()
        # Should stop after OCR + idempotency check
        harness.assert_stage_sequence(["OCR"])


class TestValidationErrors:
    """Tests for validation failure handling."""

    def test_validation_failure_creates_partial_result(self):
        """When validation fails on some fields, partial data is preserved."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE", quality=0.8)
        harness.parser.set_parse_result(
            merchant="Store",
            total=10.0,
            date="",  # Missing date
            items=[]
        )

        # Date validation fails, but preserve partial
        from tests.harness.fakes.fake_validation_service import ValidationField
        harness.validator.field_passes(ValidationField.MERCHANT)
        harness.validator.field_fails(ValidationField.DATE, "Date is required")
        harness.validator.field_passes(ValidationField.TOTAL)
        harness.validator.set_preserve_partial(True)

        result = harness.run("receipt.jpg")

        # Should save partial result
        assert result.status == PipelineStatus.PARTIAL
        assert result.receipt_data["merchant_name"] == "Store"
        harness.assert_persisted_once()

    def test_critical_validation_failure_no_preserve(self):
        """When validation fails critically and no partial preservation."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE", quality=0.8)
        harness.parser.set_parse_result(merchant="", total=None)  # Bad data

        from tests.harness.fakes.fake_validation_service import ValidationField
        harness.validator.all_fields_fail("Critical error")
        harness.validator.set_preserve_partial(False)

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED
        harness.assert_not_persisted()


class TestOCRFailures:
    """Tests for OCR failure scenarios."""

    def test_ocr_failure_fails_pipeline(self):
        """When OCR fails completely, pipeline fails."""
        harness = PipelineTestHarness()

        harness.ocr.set_should_fail(OCRError("Image unreadable"))
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED
        assert "OCR" in result.error_message or result.error_message is not None
        harness.assert_not_persisted()
        # 3 OCR retry attempts tracked
        harness.assert_stage_sequence(["OCR", "OCR", "OCR"])


class TestPersistenceFailures:
    """Tests for database/persistence failure scenarios."""

    def test_repository_failure_fails_pipeline(self):
        """When save fails, pipeline fails."""
        from core.exceptions import StorageError

        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.8)
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()
        harness.repository.set_should_fail_on("save", StorageError("DB connection lost"))

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED
        harness.assert_not_persisted()


class TestHarnessDiagnostics:
    """Tests demonstrating diagnostic capabilities."""

    def test_get_full_report(self):
        """Harness provides detailed execution report."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.8)
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")

        report = harness.get_full_report()

        assert report["status"] == "success"
        assert len(report["stages"]) == 4
        assert report["repository_saves"] == 1
        assert report["retry_count"] == 0

    def test_stage_timing(self):
        """Can retrieve timing for specific stages."""
        harness = PipelineTestHarness()

        # Add some latency to OCR
        harness.ocr.set_text_for_image("receipt.jpg", "STORE", quality=0.8)
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")

        # Verify we can get timing (actual values depend on execution)
        ocr_time = harness.get_stage_timing("OCR")
        assert ocr_time is not None
        assert ocr_time >= 0


class TestComplexScenarios:
    """Complex multi-stage interaction tests."""

    def test_full_flow_with_fallback_and_retry(self):
        """Most complex scenario: low OCR quality → fallback → parse failure → retry → success."""
        harness = PipelineTestHarness(use_fake_retry=True)

        # Low quality primary OCR
        harness.ocr.set_text_for_image("receipt.jpg", "BLURRY", quality=0.2)
        harness.ocr.set_fallback_output(
            "receipt.jpg",
            "RESTAURANT\nDinner $45.00\nTax $4.50\nTotal $49.50",
            quality=0.88
        )

        # Parser fails once on date extraction, then succeeds
        harness.parser.set_parse_fails_with(ValueError("Cannot parse date"))
        harness.parser.set_parse_result(
            merchant="Restaurant",
            total=49.50,
            date="2024-03-15",
            items=[{"description": "Dinner", "price": 45.00}]
        )

        harness.retry.set_succeed_on_attempt(2)

        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_data["total_amount"] == 49.50
        assert result.receipt_data["receipt_date"] == "2024-03-15"
        harness.assert_ocr_used_fallback()
        harness.assert_retry_count(1)
        harness.assert_persisted_once()

        # Verify full sequence (PARSING stage includes retry internally)
        harness.assert_stage_sequence([
            "OCR",      # Primary (low quality)
            "OCR",      # Fallback (high quality)
            "PARSING",  # Includes retry attempts internally
            "VALIDATION",
            "PERSISTENCE"
        ])

        # Verify retry happened within parsing stage
        harness.assert_retry_count(1)
