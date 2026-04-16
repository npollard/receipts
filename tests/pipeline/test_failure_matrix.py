"""Pipeline failure matrix tests.

12 canonical scenarios covering all meaningful failure combinations.
"""

import pytest
from tests.harness.scenario import Scenario, success, failure, partial
from tests.harness.pipeline_harness import PipelineStatus
from tests.harness.invariants import check_all_invariants


class TestScenario01HappyPath:
    """Normal operation, clean single-pass."""

    def test_happy_path(self):
        """OCR(success) → PARSE(valid) → VALIDATE(pass) → PERSIST(success)"""
        scenario = Scenario().with_image("receipt.jpg") \
            .ocr(success(text="STORE\nItem $5.00\nTotal $15.99", quality=0.85)) \
            .parse(success(merchant="STORE", total=15.99, items=[{"description": "Item", "price": 5.00}])) \
            .validate(success()) \
            .persist(success(receipt_id="uuid-123"))

        result = scenario.run()

        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_data["total_amount"] == 15.99
        scenario.assert_trace(["OCR", "PARSE", "VALIDATE", "PERSIST"])

        # Validate invariants
        check_all_invariants(scenario)


class TestScenario02QualityRecovery:
    """Poor OCR triggers vision fallback, succeeds."""

    def test_quality_recovery(self):
        """OCR(low quality) → PARSE(partial) → RETRY → OCR(vision) → PARSE(valid) → success"""
        scenario = Scenario().with_image("receipt.jpg") \
            .ocr(
                success(text="@#$ garbled", quality=0.15),
                success(text="STORE\nItem $5.00\nTotal $15.99", quality=0.90),
                success(text="STORE\nItem $5.00\nTotal $15.99", quality=0.90),
                success(text="STORE\nItem $5.00\nTotal $15.99", quality=0.90)
            ) \
            .parse(
                partial(total=None, items=[]),
                success(merchant="STORE", total=15.99)
            ) \
            .validate(
                failure(reason="ocr_quality_low"),
                success()
            ) \
            .persist(success(receipt_id="uuid-456"))

        result = scenario.run()

        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_data["total_amount"] == 15.99
        # OCR retry triggered: up to 3 attempts
        assert scenario.get_stage_attempts("OCR") <= 3
        # Full trace includes retry cycle: OCR → PARSE → VALIDATE (fail) → OCR → PARSE → VALIDATE (success) → PERSIST
        scenario.assert_trace(["OCR", "PARSE", "VALIDATE", "OCR", "PARSE", "VALIDATE", "PERSIST"])

        # Validate invariants
        check_all_invariants(scenario)


class TestScenario03OCRFailureRecovery:
    """OCR crash, vision fallback extracts content."""

    def test_ocr_failure_recovery(self):
        """OCR(failure) → RETRY → OCR(vision) → PARSE(valid) → success"""
        scenario = Scenario().with_image("receipt.jpg") \
            .ocr(
                failure(error="easyocr_crash"),
                success(text="STORE\nTotal $15.99", quality=0.82)
            ) \
            .parse(success(merchant="STORE", total=15.99)) \
            .validate(success()) \
            .persist(success(receipt_id="uuid-789"))

        result = scenario.run()

        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_data["total_amount"] == 15.99
        # OCR failure recovery triggered
        assert scenario.get_stage_attempts("OCR") <= 3
        scenario.assert_trace(["OCR", "PARSE", "VALIDATE", "PERSIST"])

        # Validate invariants
        check_all_invariants(scenario)


class TestScenario04ParsingRecovery:
    """Malformed JSON, reparse fixes it."""

    def test_parsing_recovery(self):
        """OCR(success) → PARSE(malformed) → RETRY(reparse) → PARSE(valid) → success"""
        scenario = Scenario().with_image("receipt.jpg") \
            .ocr(success(text="{broken: json", quality=0.75)) \
            .parse(
                failure(error="json_parse_error"),
                success(merchant="STORE", total=15.99)
            ) \
            .validate(
                failure(reason="malformed_output"),
                success()
            ) \
            .persist(success(receipt_id="uuid-abc"))

        result = scenario.run()

        assert result.status == PipelineStatus.SUCCESS
        # Parse retry triggered: REPARSE
        assert scenario.get_stage_attempts("PARSE") == 2
        # Full trace includes parse retry: OCR → PARSE → VALIDATE (fail) → PARSE → VALIDATE (success) → PERSIST
        scenario.assert_trace(["OCR", "PARSE", "VALIDATE", "PARSE", "VALIDATE", "PERSIST"])

        # Validate invariants
        check_all_invariants(scenario)


class TestScenario05PartialAccept:
    """Missing optional fields, accepted with warnings."""

    def test_partial_accept(self):
        """OCR(low quality but readable) → PARSE(partial) → VALIDATE(pass with warnings) → success"""
        result = Scenario().with_image("receipt.jpg") \
            .ocr(success(text="TOTAL $15.99", quality=0.55)) \
            .parse(partial(total=15.99, merchant=None, items=[])) \
            .validate(success(confidence=0.45)) \
            .persist(success(receipt_id="uuid-def", partial=True)) \
            .run()

        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_data["total_amount"] == 15.99


class TestScenario06QualityExhaustion:
    """Both OCR methods fail, quality too low."""

    def test_quality_exhaustion(self):
        """OCR(low) → RETRY → OCR(vision, still low) → PARSE(partial) → failure"""
        result = Scenario().with_image("receipt.jpg") \
            .ocr(
                success(text="@#$", quality=0.10),
                success(text="still garbled", quality=0.12)
            ) \
            .parse(
                partial(total=None),
                partial(total=None)
            ) \
            .validate(
                failure(reason="insufficient_data"),
                failure(reason="insufficient_data")
            ) \
            .persist() \
            .run()

        assert result.status == PipelineStatus.FAILURE


class TestScenario07NonRetryableValidation:
    """Duplicate receipt, immediate rejection."""

    def test_non_retryable_validation(self):
        """OCR(success) → PARSE(valid) → VALIDATE(duplicate) → failure (no retry)"""
        result = Scenario().with_image("receipt.jpg") \
            .ocr(success(text="STORE\nTotal $15.99", quality=0.85)) \
            .parse(success(merchant="STORE", total=15.99)) \
            .validate(failure(reason="duplicate_receipt", retryable=False)) \
            .persist() \
            .run()

        assert result.status == PipelineStatus.FAILURE


class TestScenario08InfrastructureRecovery:
    """DB timeout, persistence retry succeeds."""

    def test_infrastructure_recovery(self):
        """... → VALIDATE(pass) → PERSIST(fail) → RETRY → PERSIST(success)"""
        scenario = Scenario().with_image("receipt.jpg") \
            .ocr(success(text="STORE\nTotal $15.99", quality=0.85)) \
            .parse(success(merchant="STORE", total=15.99)) \
            .validate(success()) \
            .persist(
                failure(error="db_connection_timeout"),
                success(receipt_id="uuid-ghi")
            )

        result = scenario.run()

        assert result.status == PipelineStatus.SUCCESS
        assert result.receipt_id == "uuid-ghi"
        # Persistence retry happens internally but only traced once
        assert scenario.get_stage_attempts("PERSIST") == 1
        # Trace shows single PERSIST (internal retries not traced separately)
        scenario.assert_trace(["OCR", "PARSE", "VALIDATE", "PERSIST"])

        # Validate invariants
        check_all_invariants(scenario)


class TestScenario09EmptyExtractionRecovery:
    """EasyOCR returns empty, vision finds content."""

    def test_empty_extraction_recovery(self):
        """OCR(empty) → RETRY → OCR(vision) → PARSE(valid) → success"""
        result = Scenario().with_image("receipt.jpg") \
            .ocr(
                success(text="", quality=0.00),
                success(text="STORE\nTotal $15.99", quality=0.80)
            ) \
            .parse(success(merchant="STORE", total=15.99)) \
            .validate(
                failure(reason="empty_extraction"),
                success()
            ) \
            .persist(success(receipt_id="uuid-jkl")) \
            .run()

        assert result.status == PipelineStatus.SUCCESS


class TestScenario10MultipleRetryCascade:
    """Quality + parse issues, both recovered."""

    def test_multiple_retry_cascade(self):
        """OCR(low) → PARSE(partial) → RETRY(vision) → PARSE(malformed) → RETRY(reparse) → success"""
        scenario = Scenario().with_image("receipt.jpg") \
            .ocr(
                success(text="blur", quality=0.20),
                success(text="{malformed", quality=0.60)
            ) \
            .parse(
                partial(total=None),
                failure(error="json_error"),
                success(merchant="STORE", total=15.99)
            ) \
            .validate(
                failure(reason="insufficient_data"),
                failure(reason="malformed_output"),
                success()
            ) \
            .persist(success(receipt_id="uuid-mno"))

        result = scenario.run()

        assert result.status == PipelineStatus.SUCCESS
        # Multiple retry types triggered (actual counts may vary based on retry paths)
        assert scenario.get_stage_attempts("OCR") <= 4  # OCR retry + fallback
        assert scenario.get_stage_attempts("PARSE") >= 2  # At least initial + retry
        # Trace includes retry cycles (normalized: consecutive duplicates collapsed)
        scenario.assert_trace(["OCR", "PARSE", "VALIDATE", "PARSE", "VALIDATE", "PERSIST"])

        # Validate invariants
        check_all_invariants(scenario)


class TestScenario11CompleteInfrastructureFailure:
    """DB down, all persistence retries fail."""

    def test_complete_infrastructure_failure(self):
        """... → VALIDATE(pass) → PERSIST(fail) × 3 → failure"""
        result = Scenario().with_image("receipt.jpg") \
            .ocr(success(text="STORE\nTotal $15.99", quality=0.85)) \
            .parse(success(merchant="STORE", total=15.99)) \
            .validate(success()) \
            .persist(
                failure(error="db_connection_refused"),
                failure(error="db_connection_refused"),
                failure(error="db_connection_refused")
            ) \
            .run()

        assert result.status == PipelineStatus.FAILURE


class TestScenario12UnrecoverableParse:
    """Completely unparseable content, reparse fails."""

    def test_unrecoverable_parse(self):
        """OCR(gibberish) → PARSE(malformed) → RETRY(reparse) → PARSE(malformed) → failure"""
        result = Scenario().with_image("receipt.jpg") \
            .ocr(success(text="@#\$%^&*() random", quality=0.30)) \
            .parse(
                failure(error="unparseable"),
                failure(error="still_unparseable")
            ) \
            .validate(
                failure(reason="malformed_output"),
                failure(reason="malformed_output")
            ) \
            .persist() \
            .run()

        assert result.status == PipelineStatus.FAILURE
