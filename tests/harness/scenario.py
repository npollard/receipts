"""Lightweight Scenario DSL for pipeline testing.

Provides fluent builder interface on top of PipelineTestHarness.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from tests.harness.pipeline_harness import PipelineTestHarness, PipelineStatus
from tests.harness.fakes.fake_ocr_service import OCROutput


@dataclass
class Attempt:
    """Single stage attempt configuration."""
    status: str
    params: Dict[str, Any] = field(default_factory=dict)


class StageBuilder:
    """Builder for a single stage with multiple attempts."""

    def __init__(self, name: str):
        self.name = name
        self.attempts: List[Attempt] = []

    def success(self, **params) -> 'StageBuilder':
        """Add successful attempt."""
        self.attempts.append(Attempt("success", params))
        return self

    def failure(self, error: str, **params) -> 'StageBuilder':
        """Add failed attempt."""
        params["error"] = error
        self.attempts.append(Attempt("failure", params))
        return self

    def partial(self, **params) -> 'StageBuilder':
        """Add partial success attempt."""
        self.attempts.append(Attempt("partial", params))
        return self

    def __len__(self) -> int:
        return len(self.attempts)


class Scenario:
    """Fluent scenario builder for pipeline tests.

    Example:
        scenario = Scenario()
            .ocr(success(quality=0.85, text="STORE $10"))
            .parse(success(merchant="Store", total=10.0))
            .validate(success())
            .persist(success(receipt_id="uuid-123"))
    """

    def __init__(self):
        self._ocr = StageBuilder("OCR")
        self._parse = StageBuilder("PARSE")
        self._validate = StageBuilder("VALIDATE")
        self._persist = StageBuilder("PERSIST")
        self._image_path = "receipt.jpg"
        self._harness: Optional[PipelineTestHarness] = None

    def with_image(self, path: str) -> 'Scenario':
        """Set image path for scenario."""
        self._image_path = path
        return self

    def ocr(self, *attempts: Attempt) -> 'Scenario':
        """Configure OCR stage attempts."""
        self._ocr.attempts = list(attempts)
        return self

    def parse(self, *attempts: Attempt) -> 'Scenario':
        """Configure parse stage attempts."""
        self._parse.attempts = list(attempts)
        return self

    def validate(self, *attempts: Attempt) -> 'Scenario':
        """Configure validation stage attempts."""
        self._validate.attempts = list(attempts)
        return self

    def persist(self, *attempts: Attempt) -> 'Scenario':
        """Configure persistence stage attempts."""
        self._persist.attempts = list(attempts)
        return self

    def run(self) -> Any:
        """Execute scenario and return result."""
        harness = PipelineTestHarness()

        # Configure OCR using sequence for multiple attempts
        if self._ocr.attempts:
            sequence = []
            for attempt in self._ocr.attempts:
                if attempt.status == "success":
                    sequence.append(OCROutput(
                        text=attempt.params.get("text", ""),
                        quality_score=attempt.params.get("quality", 0.5),
                        method=attempt.params.get("method", "easyocr")
                    ))
                elif attempt.status == "failure":
                    sequence.append(Exception(attempt.params.get("error", "ocr_error")))
            if sequence:
                harness.ocr.set_sequence(sequence)

        # Configure parser using sequence for multiple attempts
        if self._parse.attempts:
            from tests.harness.fakes.fake_receipt_parser import ParserOutput
            sequence = []
            for attempt in self._parse.attempts:
                if attempt.status == "success":
                    sequence.append(ParserOutput(
                        merchant=attempt.params.get("merchant", "Store"),
                        total=attempt.params.get("total", 0.0),
                        date=attempt.params.get("date", ""),
                        items=attempt.params.get("items", [])
                    ))
                elif attempt.status == "partial":
                    sequence.append(ParserOutput(
                        merchant=attempt.params.get("merchant"),
                        total=attempt.params.get("total"),
                        date=attempt.params.get("date", ""),
                        items=attempt.params.get("items", [])
                    ))
                elif attempt.status == "failure":
                    sequence.append(Exception(attempt.params.get("error", "parse_error")))
            if sequence:
                harness.parser.set_sequence(sequence)

        # Configure validator using sequence for multiple attempts
        if self._validate.attempts:
            from tests.harness.fakes.fake_validation_service import ValidationResult, FieldValidation, ValidationField
            sequence = []
            for attempt in self._validate.attempts:
                if attempt.status == "success":
                    # Build field_results for all fields passing
                    field_results = {
                        field.value: FieldValidation(field=field.value, is_valid=True)
                        for field in ValidationField
                    }
                    sequence.append(ValidationResult(
                        is_valid=True,
                        field_results=field_results,
                        errors=[],
                        warnings=[]
                    ))
                elif attempt.status == "failure":
                    # Build field_results with TOTAL failing
                    field_results = {
                        field.value: FieldValidation(
                            field=field.value,
                            is_valid=(field != ValidationField.TOTAL),
                            error_message=None if field != ValidationField.TOTAL else attempt.params.get("reason", "validation_error")
                        )
                        for field in ValidationField
                    }
                    # Map error types to retryable and preserve_partial
                    reason = attempt.params.get("reason", "validation_error")
                    # duplicate_receipt → terminal failure (not retryable, no partial)
                    # insufficient_data, malformed_output, ocr_quality_low, json_parse_error, empty_extraction → retryable
                    retryable = reason in ("insufficient_data", "malformed_output", "ocr_quality_low", "json_parse_error", "empty_extraction")
                    preserve_partial = reason != "duplicate_receipt"
                    sequence.append(ValidationResult(
                        is_valid=False,
                        field_results=field_results,
                        errors=[reason],
                        warnings=[],
                        retryable=retryable,
                        preserve_partial=preserve_partial
                    ))
            if sequence:
                harness.validator.set_sequence(sequence)

        # Configure persistence via repository
        # Track failures - clear on final success if there were previous failures
        failure_count = sum(1 for a in self._persist.attempts if a.status == "failure")
        has_success = any(a.status == "success" for a in self._persist.attempts)

        if failure_count > 0:
            # Will fail 'failure_count' times then succeed
            harness.repository.set_should_fail_on(
                "save",
                Exception(self._persist.attempts[0].params.get("error", "db_error"))
            )
            # Store failure count in repository for internal tracking
            harness.repository._failure_count = failure_count
            harness.repository._success_after_failures = has_success

        # Configure receipt_id for success attempts
        if has_success:
            success_attempt = next(a for a in self._persist.attempts if a.status == "success")
            receipt_id = success_attempt.params.get("receipt_id")
            if receipt_id:
                harness.repository.should_succeed(receipt_id)

        self._harness = harness
        return harness.run(self._image_path)

    def __call__(self) -> Any:
        """Allow scenario to be called directly."""
        return self.run()

    def assert_trace(self, expected_stages: List[str]) -> None:
        """Validate exact stage execution sequence.

        Args:
            expected_stages: List of stage names in expected order
                e.g., ["OCR", "OCR", "PARSE", "VALIDATE", "PERSIST"]

        Raises:
            AssertionError: If actual sequence doesn't match expected
        """
        if self._harness is None:
            raise RuntimeError("Scenario must be run before assert_trace")

        self._harness.assert_stage_sequence(expected_stages)

    def get_stage_attempts(self, stage: str) -> int:
        """Get number of attempts for a specific stage.

        Args:
            stage: Stage name (e.g., "OCR", "PARSE")

        Returns:
            Number of times stage was attempted
        """
        if self._harness is None:
            raise RuntimeError("Scenario must be run before get_stage_attempts")

        return self._harness.get_stage_attempts(stage)


# Convenience functions for attempt creation

def success(**params) -> Attempt:
    """Create successful attempt."""
    return Attempt("success", params)


def failure(error: str = None, reason: str = None, retryable: bool = True, **params) -> Attempt:
    """Create failed attempt.

    Args:
        error: Error message/code (primary identifier)
        reason: Alternative to error (mapped internally)
        retryable: Whether this failure can be retried
        **params: Additional metadata

    Returns:
        Attempt configured for failure
    """
    # Normalize: reason → error
    error_value = error or reason or "unknown_error"
    params["error"] = error_value
    if reason:
        params["reason"] = reason
    params["retryable"] = retryable
    return Attempt("failure", params)


def partial(**params) -> Attempt:
    """Create partial success attempt."""
    return Attempt("partial", params)
