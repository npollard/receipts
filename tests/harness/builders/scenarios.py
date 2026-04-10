"""Pre-configured test scenarios for common pipeline paths.

Scenarios encapsulate complex harness configurations into readable,
reusable functions. Tests should read like specifications.

Example:
    >>> harness = make_happy_path()
    >>> result = harness.run("receipt.jpg")
    >>> assert result.status == PipelineStatus.SUCCESS
"""

from typing import Optional, List
from tests.harness.pipeline_harness import PipelineTestHarness, PipelineStatus
from tests.harness.builders import ReceiptBuilder
from tests.harness.fakes.fake_validation_service import ValidationField
from core.exceptions import OCRError


def make_happy_path(
    merchant: str = "Grocery Store",
    total: float = 25.50,
    ocr_quality: float = 0.85
) -> PipelineTestHarness:
    """Standard successful processing with high-quality OCR.

    Scenario: Clean receipt, good OCR, valid data, no retries needed.

    Returns:
        Configured harness ready to run
    """
    harness = PipelineTestHarness()

    harness.ocr.set_text_for_image(
        "receipt.jpg",
        f"{merchant}\nTotal ${total}",
        quality=ocr_quality
    )

    harness.parser.set_parse_result(
        merchant=merchant,
        total=total,
        date="2024-03-15",
        items=[
            {"description": "Item 1", "price": total * 0.6},
            {"description": "Item 2", "price": total * 0.4}
        ]
    )

    harness.validator.all_fields_pass()

    return harness


def make_easyocr_fail_then_vision_success(
    merchant: str = "Restaurant",
    total: float = 45.00
) -> PipelineTestHarness:
    """Low-quality EasyOCR triggers Vision fallback, then succeeds.

    Scenario: Primary OCR quality < 0.25, fallback succeeds.

    Returns:
        Configured harness ready to run
    """
    harness = PipelineTestHarness()

    # Primary OCR: low quality
    harness.ocr.set_text_for_image(
        "receipt.jpg",
        "BLURRY TEXT...",
        quality=0.15
    )

    # Fallback: high quality
    harness.ocr.set_fallback_output(
        "receipt.jpg",
        f"{merchant}\nDinner ${total * 0.9:.2f}\nTax ${total * 0.1:.2f}\nTotal ${total}",
        quality=0.88
    )

    harness.parser.set_parse_result(
        merchant=merchant,
        total=total,
        date="2024-03-15",
        items=[{"description": "Dinner", "price": total * 0.9}],
        tax=total * 0.1
    )

    harness.validator.all_fields_pass()

    return harness


def make_validation_failure_then_retry_success(
    merchant: str = "Store",
    total: float = 10.00
) -> PipelineTestHarness:
    """Validation fails initially, LLM self-correction fixes it.

    Scenario: Parser misses date field, validation catches it,
    retry with LLM_SELF_CORRECTION strategy adds date.

    Returns:
        Configured harness ready to run
    """
    harness = PipelineTestHarness(use_fake_retry=True)

    harness.ocr.set_text_for_image("receipt.jpg", f"{merchant} ${total}", quality=0.8)

    # First parse: fails with error (triggers retry)
    harness.parser.set_parse_fails_with(ValueError("Missing required field: date"))

    # Second parse: succeeds (retry success)
    harness.parser.set_parse_result(
        merchant=merchant,
        total=total,
        date="2024-01-15",
        items=[{"description": "Item", "price": total}]
    )

    # All validation passes on successful parse
    harness.validator.all_fields_pass()

    # Retry configured
    harness.retry.set_succeed_on_attempt(2)
    harness.retry.set_strategy_for_error(ValueError, "LLM_SELF_CORRECTION")

    return harness


def make_duplicate_detected(
    merchant: str = "Original Store",
    total: float = 25.00
) -> PipelineTestHarness:
    """Same image already processed, returns existing receipt.

    Scenario: Idempotency check finds existing receipt by image hash,
    pipeline returns duplicate without re-processing.

    Returns:
        Configured harness with seeded receipt
    """
    harness = PipelineTestHarness()

    # Seed existing receipt with matching hash
    existing = (ReceiptBuilder()
        .with_image_hash("hash_receipt")  # Matches "receipt.jpg"
        .with_merchant(merchant)
        .with_total(total)
        .build())
    harness.with_seeded_receipt(existing)

    # OCR configured but won't be used past idempotency check
    harness.ocr.set_text_for_image("receipt.jpg", "ANY TEXT")

    return harness


def make_partial_save_with_validation_errors(
    merchant: str = "Partial Store",
    total: float = 30.00
) -> PipelineTestHarness:
    """Some fields invalid, but partial data preserved.

    Scenario: Date and tax missing, but merchant/total valid.
    Partial result saved with preserve_partial=True.

    Returns:
        Configured harness ready to run
    """
    harness = PipelineTestHarness()

    harness.ocr.set_text_for_image("receipt.jpg", f"{merchant} ${total}", quality=0.8)

    harness.parser.set_parse_result(
        merchant=merchant,
        total=total,
        date="",  # Missing
        items=[],
        tax=None  # Missing
    )

    # Validation: merchant/total pass, date/tax fail
    harness.validator.field_passes(ValidationField.MERCHANT)
    harness.validator.field_passes(ValidationField.TOTAL)
    harness.validator.field_fails(ValidationField.DATE, "Missing date")
    harness.validator.set_preserve_partial(True)

    return harness


def make_full_failure_ocr_error() -> PipelineTestHarness:
    """OCR completely fails, pipeline aborts.

    Scenario: Image unreadable, OCRError raised, no persistence.

    Returns:
        Configured harness ready to run
    """
    harness = PipelineTestHarness()

    harness.ocr.set_should_fail(OCRError("Image corrupt or unreadable"))

    # Parser configured but won't be reached
    harness.parser.set_parse_result(merchant="Never Used", total=0.0)

    return harness


def make_full_failure_validation_critical() -> PipelineTestHarness:
    """Critical validation failures, no partial preservation.

    Scenario: All fields invalid, preserve_partial=False,
    pipeline fails without saving.

    Returns:
        Configured harness ready to run
    """
    harness = PipelineTestHarness()

    harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.8)

    harness.parser.set_parse_result(
        merchant="",
        total=None,
        date="invalid-date"
    )

    harness.validator.all_fields_fail("Critical validation error")
    harness.validator.set_preserve_partial(False)

    return harness


def make_retry_exhausted_all_attempts_fail(
    max_retries: int = 3
) -> PipelineTestHarness:
    """All retry attempts exhausted, parser never succeeds.

    Scenario: Parser fails consistently, retry gives up, pipeline fails.

    Args:
        max_retries: Number of retry attempts before giving up

    Returns:
        Configured harness ready to run
    """
    harness = PipelineTestHarness(use_fake_retry=True)

    harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.8)

    # Parser always fails
    harness.parser.set_parse_fails_with(ValueError("Unparseable content"))

    # Retry never succeeds
    harness.retry.set_max_retries(max_retries)
    harness.retry.set_succeed_on_attempt(999)  # Never

    return harness


def make_complex_fallback_and_retry(
    merchant: str = "Complex Store",
    total: float = 55.00
) -> PipelineTestHarness:
    """Most complex scenario: low OCR quality + parse failure + retry success.

    Scenario:
    1. EasyOCR returns low quality (0.2)
    2. Vision fallback succeeds with good text
    3. Parser fails on first attempt (missing field)
    4. Retry with LLM_SELF_CORRECTION succeeds

    Returns:
        Configured harness ready to run
    """
    harness = PipelineTestHarness(use_fake_retry=True)

    # Step 1 & 2: OCR with fallback
    harness.ocr.set_text_for_image("receipt.jpg", "BLURRY", quality=0.20)
    harness.ocr.set_fallback_output(
        "receipt.jpg",
        f"{merchant}\nItem ${total * 0.8:.2f}\nTax ${total * 0.2:.2f}\nTotal ${total}",
        quality=0.90
    )

    # Step 3: Parser fails first attempt
    harness.parser.set_parse_fails_with(ValueError("Missing required field: date"))

    # Step 4: Parser succeeds on retry
    harness.parser.set_parse_result(
        merchant=merchant,
        total=total,
        date="2024-03-15",
        items=[{"description": "Item", "price": total * 0.8}],
        tax=total * 0.2
    )

    # Validation passes on retry result
    harness.validator.all_fields_pass()

    # Retry configured
    harness.retry.set_succeed_on_attempt(2)
    harness.retry.set_strategy_for_error(ValueError, "LLM_SELF_CORRECTION")

    return harness


class ScenarioBuilder:
    """Fluent builder for custom scenarios.

    Example:
        >>> scenario = (ScenarioBuilder()
        ...     .with_high_quality_ocr()
        ...     .with_parse_retry_on_attempt(2)
        ...     .with_validation_passing()
        ...     .build())
    """

    def __init__(self):
        self._harness = PipelineTestHarness()
        self._image_path = "receipt.jpg"

    def with_high_quality_ocr(self, text: str = "STORE $10") -> "ScenarioBuilder":
        """Configure high-quality EasyOCR (no fallback)."""
        self._harness.ocr.set_text_for_image(self._image_path, text, quality=0.85)
        return self

    def with_low_quality_ocr_triggering_fallback(
        self,
        primary_text: str = "BLURRY",
        fallback_text: str = "STORE $10"
    ) -> "ScenarioBuilder":
        """Configure low-quality OCR with Vision fallback."""
        self._harness.ocr.set_text_for_image(self._image_path, primary_text, quality=0.20)
        self._harness.ocr.set_fallback_output(self._image_path, fallback_text, quality=0.88)
        return self

    def with_ocr_failing(self, error: Exception = OCRError("Failed")) -> "ScenarioBuilder":
        """Configure OCR to fail."""
        self._harness.ocr.set_should_fail(error)
        return self

    def with_parsed_receipt(
        self,
        merchant: str = "Store",
        total: float = 10.0,
        **kwargs
    ) -> "ScenarioBuilder":
        """Configure successful parse result."""
        self._harness.parser.set_parse_result(merchant=merchant, total=total, **kwargs)
        return self

    def with_parse_retry_on_attempt(self, attempt: int) -> "ScenarioBuilder":
        """Configure parser to need retry."""
        self._harness = PipelineTestHarness(use_fake_retry=True)
        # First fails
        self._harness.parser.set_parse_fails_with(ValueError("Retry needed"))
        # Second succeeds (will be configured by user or default)
        self._harness.parser.set_parse_result(merchant="Store", total=10.0)
        self._harness.retry.set_succeed_on_attempt(attempt)
        return self

    def with_validation_passing(self) -> "ScenarioBuilder":
        """Configure all validations to pass."""
        self._harness.validator.all_fields_pass()
        return self

    def with_validation_failing_on(
        self,
        field: str,
        preserve_partial: bool = True
    ) -> "ScenarioBuilder":
        """Configure specific field to fail validation."""
        field_enum = ValidationField(field)
        self._harness.validator.field_fails(field_enum)
        self._harness.validator.set_preserve_partial(preserve_partial)
        return self

    def with_seeded_receipt(self, receipt_dto) -> "ScenarioBuilder":
        """Pre-populate repository with receipt."""
        self._harness.with_seeded_receipt(receipt_dto)
        return self

    def build(self) -> PipelineTestHarness:
        """Return configured harness."""
        return self._harness
