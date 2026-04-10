"""PipelineTestHarness - High-level orchestration for deterministic pipeline testing.

Wires together all fake components and provides a fluent API for testing
complete receipt processing flows with full observability.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
import time

from tests.harness.fakes import (
    FakeOCRService,
    FakeReceiptParser,
    FakeValidationService,
    FakeRepository,
    FakeRetryService,
    OCROutput,
    ParserOutput,
    ValidationResult,
)
from tests.harness.fakes.fake_component import CallRecord


class PipelineStage(Enum):
    """Stages in the receipt processing pipeline."""
    OCR = auto()
    PARSING = auto()
    VALIDATION = auto()
    RETRY = auto()
    PERSISTENCE = auto()


class PipelineStatus(Enum):
    """Final status of pipeline execution."""
    SUCCESS = "success"
    FAILED = "failed"
    DUPLICATE = "duplicate"
    PARTIAL = "partial"


@dataclass
class StageTransition:
    """Record of a stage transition."""
    stage: PipelineStage
    entered_at: datetime
    exited_at: Optional[datetime] = None
    input_data: Any = None
    output_data: Any = None
    exception: Optional[Exception] = None


@dataclass
class PipelineResult:
    """Complete result of pipeline execution."""
    status: PipelineStatus
    receipt_data: Optional[Dict[str, Any]] = None
    receipt_id: Optional[str] = None
    error_message: Optional[str] = None
    stages: List[StageTransition] = field(default_factory=list)
    duration_ms: float = 0.0
    retry_count: int = 0
    ocr_method: Optional[str] = None
    was_duplicate: bool = False
    preserved_partial: bool = False


@dataclass
class PipelineState:
    """Mutable state tracked during pipeline execution."""
    image_path: str = ""
    ocr_text: str = ""
    ocr_quality: float = 0.0
    parsed_data: Optional[Dict[str, Any]] = None
    validation_result: Optional[ValidationResult] = None
    saved_receipt: Any = None
    current_stage: Optional[PipelineStage] = None
    transitions: List[StageTransition] = field(default_factory=list)


class PipelineTestHarness:
    """High-level harness for testing complete receipt processing pipelines.

    Wires together all fake components and orchestrates execution while
    capturing state transitions for assertions.

    Example:
        >>> harness = PipelineTestHarness()
        >>> harness.ocr.set_text_for_image("receipt.jpg", "STORE $10")
        >>> harness.parser.set_parse_result(merchant="Store", total=10.0)
        >>> harness.validator.all_fields_pass()
        >>>
        >>> result = harness.run("receipt.jpg")
        >>>
        >>> harness.assert_persisted_once()
        >>> harness.assert_stage_sequence(["OCR", "PARSING", "VALIDATION", "PERSISTENCE"])
    """

    def __init__(self, use_fake_retry: bool = True):
        """Initialize harness with all fake components.

        Args:
            use_fake_retry: If True, use FakeRetryService; otherwise use real RetryService
        """
        self.ocr = FakeOCRService()
        self.parser = FakeReceiptParser()
        self.validator = FakeValidationService()
        self.repository = FakeRepository()

        if use_fake_retry:
            self.retry = FakeRetryService()
            self._retry_is_fake = True
        else:
            # Import real retry service
            from services.retry_service import RetryService
            self.retry = RetryService(max_retries=3)
            self._retry_is_fake = False

        self._state = PipelineState()
        self._last_result: Optional[PipelineResult] = None
        self._user_id: str = "test_user"
        self._image_hashes: Dict[str, str] = {}  # path -> hash

    def with_user(self, user_id: str) -> "PipelineTestHarness":
        """Set the user ID for this pipeline run."""
        self._user_id = user_id
        return self

    def with_seeded_receipt(self, receipt_dto: Any) -> "PipelineTestHarness":
        """Pre-populate repository with a receipt (for idempotency tests)."""
        self.repository.seed_receipt(receipt_dto)
        return self

    def with_seeded_user(self, email: str) -> "PipelineTestHarness":
        """Pre-populate repository with a user."""
        user = self.repository.get_or_create_user(email)
        self._user_id = user.id
        return self

    def _start_stage(self, stage: PipelineStage, input_data: Any = None) -> StageTransition:
        """Begin tracking a pipeline stage."""
        transition = StageTransition(
            stage=stage,
            entered_at=datetime.now(),
            input_data=input_data
        )
        self._state.current_stage = stage
        self._state.transitions.append(transition)
        return transition

    def _end_stage(self, stage: PipelineStage, output_data: Any = None,
                   exception: Optional[Exception] = None) -> None:
        """Complete tracking of current stage."""
        for transition in reversed(self._state.transitions):
            if transition.stage == stage and transition.exited_at is None:
                transition.exited_at = datetime.now()
                transition.output_data = output_data
                transition.exception = exception
                break
        self._state.current_stage = None

    def run(self, image_path: str) -> PipelineResult:
        """Execute the full pipeline on an image.

        Args:
            image_path: Path to the receipt image

        Returns:
            PipelineResult with full execution details
        """
        start_time = time.time()
        self._state = PipelineState(image_path=image_path)

        try:
            # Stage 1: OCR
            ocr_result = self._run_ocr(image_path)
            if not ocr_result:
                return self._build_result(PipelineStatus.FAILED, "OCR failed")

            # Stage 2: Check for duplicate by image hash
            existing = self._check_idempotency(image_path)
            if existing:
                return self._build_result(
                    PipelineStatus.DUPLICATE,
                    receipt_id=existing.id,
                    was_duplicate=True
                )

            # Stage 3: Parse
            parse_result = self._run_parsing(self._state.ocr_text)
            if not parse_result:
                return self._build_result(PipelineStatus.FAILED, "Parsing failed")

            # Stage 4: Validate
            validation_result = self._run_validation(self._state.parsed_data)
            if not validation_result:
                return self._build_result(PipelineStatus.FAILED, "Validation failed")

            # Check validation result before persisting
            if not validation_result.is_valid:
                if validation_result.preserve_partial and self._state.parsed_data:
                    # Stage 5a: Persist partial result
                    receipt = self._run_persistence(self._state.parsed_data, image_path)
                    if not receipt:
                        return self._build_result(PipelineStatus.FAILED, "Persistence failed")
                    return self._build_result(
                        PipelineStatus.PARTIAL,
                        receipt_id=receipt.id if receipt else None,
                        receipt_data=self._state.parsed_data
                    )
                else:
                    # Validation failed, no partial preservation
                    return self._build_result(
                        PipelineStatus.FAILED,
                        error_message="Validation failed: " + "; ".join(validation_result.errors)
                    )

            # Stage 5: Persist successful result
            receipt = self._run_persistence(self._state.parsed_data, image_path)
            if not receipt:
                return self._build_result(PipelineStatus.FAILED, "Persistence failed")

            return self._build_result(
                PipelineStatus.SUCCESS,
                receipt_id=receipt.id if receipt else None,
                receipt_data=self._state.parsed_data
            )

        except Exception as e:
            return self._build_result(PipelineStatus.FAILED, str(e))

    def _run_ocr(self, image_path: str, max_attempts: int = 3) -> bool:
        """Execute OCR stage with optional retry support.

        Args:
            image_path: Path to image
            max_attempts: Maximum OCR attempts before giving up
        """
        last_exception = None

        for attempt in range(max_attempts):
            is_fallback = False
            transition = self._start_stage(
                PipelineStage.OCR,
                {"image_path": image_path, "type": "fallback" if is_fallback else "primary", "attempt": attempt + 1}
            )

            try:
                # Try OCR
                text = self.ocr.extract_text(image_path)
                quality = self.ocr.score_ocr_quality(text)

                self._state.ocr_text = text
                self._state.ocr_quality = quality

                # Check quality threshold for fallback
                if quality < 0.25:
                    self._end_stage(PipelineStage.OCR, {
                        "text": text,
                        "quality": quality,
                        "method": "easyocr",
                        "triggered_fallback": True
                    })

                    # Start fallback stage
                    fallback_transition = self._start_stage(
                        PipelineStage.OCR,
                        {"image_path": image_path, "type": "fallback", "attempt": attempt + 1}
                    )

                    try:
                        text = self.ocr.extract_text(image_path, use_vision_fallback=True)
                        quality = self.ocr.score_ocr_quality(text)
                        self._state.ocr_text = text
                        self._state.ocr_quality = quality
                    except Exception:
                        pass  # Use original low-quality result

                    self._end_stage(PipelineStage.OCR, {
                        "text": text,
                        "quality": quality,
                        "method": "vision"
                    })
                else:
                    self._end_stage(PipelineStage.OCR, {
                        "text": text,
                        "quality": quality,
                        "method": "easyocr"
                    })
                return True

            except Exception as e:
                last_exception = e
                self._end_stage(PipelineStage.OCR, exception=e)

                # Continue to next attempt if retries remain
                if attempt < max_attempts - 1:
                    continue
                else:
                    return False

        return False

    def _check_idempotency(self, image_path: str) -> Optional[Any]:
        """Check if receipt already exists."""
        # Use the image hash from OCR call if available, otherwise compute
        image_hash = self._image_hashes.get(image_path)
        if not image_hash:
            # Try to get from last OCR call
            ocr_calls = self.ocr.get_calls("extract_text")
            if ocr_calls:
                # The OCR call recorded the output, but we need the hash
                # Use a deterministic hash based on path for testing
                image_hash = f"hash_{Path(image_path).stem}"
            else:
                image_hash = f"hash_{Path(image_path).stem}"
            self._image_hashes[image_path] = image_hash

        existing = self.repository.find_by_image_hash(image_hash)
        return existing

    def _run_parsing(self, text: str) -> bool:
        """Execute parsing stage with optional retry."""
        transition = self._start_stage(PipelineStage.PARSING, {"text": text})

        last_successful_result = None

        def parse_operation():
            nonlocal last_successful_result
            response = self.parser.parse_text(text)
            if response.status == "success":
                last_successful_result = response.data
                return response.data
            raise ValueError(f"Parse failed: {response.error}")

        try:
            if self._retry_is_fake:
                # Use fake retry with configured behavior
                result = self.retry.execute_with_retry(parse_operation)
                # If retry "forced success", we still have the actual data from last_successful_result
                if isinstance(result, dict) and result.get("forced_success"):
                    result = last_successful_result
            else:
                # Use real retry
                result = self.retry.execute_with_retry(parse_operation)

            self._state.parsed_data = result
            self._end_stage(PipelineStage.PARSING, result)
            return True

        except Exception as e:
            self._end_stage(PipelineStage.PARSING, exception=e)
            return False

    def _run_validation(self, data: Dict[str, Any]) -> Optional[ValidationResult]:
        """Execute validation stage."""
        transition = self._start_stage(PipelineStage.VALIDATION, {"data": data})

        try:
            result = self.validator.validate_receipt(data)
            self._state.validation_result = result
            self._end_stage(PipelineStage.VALIDATION, result)
            return result

        except Exception as e:
            self._end_stage(PipelineStage.VALIDATION, exception=e)
            return None

    def _run_persistence(self, data: Dict[str, Any], image_path: str) -> Optional[Any]:
        """Execute persistence stage."""
        transition = self._start_stage(PipelineStage.PERSISTENCE, {"data": data})

        try:
            # Get image hash
            image_hash = self._image_hashes.get(image_path, f"hash_{Path(image_path).stem}")

            receipt = self.repository.save_receipt(
                user_id=self._user_id,
                image_path=image_path,
                receipt_data=data,
                image_hash=image_hash
            )

            self._state.saved_receipt = receipt
            self._end_stage(PipelineStage.PERSISTENCE, receipt)
            return receipt

        except Exception as e:
            self._end_stage(PipelineStage.PERSISTENCE, exception=e)
            return None

    def _build_result(
        self,
        status: PipelineStatus,
        error_message: Optional[str] = None,
        receipt_id: Optional[str] = None,
        receipt_data: Optional[Dict[str, Any]] = None,
        was_duplicate: bool = False
    ) -> PipelineResult:
        """Build the final PipelineResult."""
        end_time = time.time()

        # Calculate retry count from fake retry service
        retry_count = 0
        if self._retry_is_fake:
            retry_count = max(0, self.retry.get_attempt_count() - 1)

        # Get OCR method from last OCR call
        ocr_method = None
        ocr_calls = self.ocr.get_calls("extract_text")
        if ocr_calls:
            output = ocr_calls[-1].result
            if isinstance(output, OCROutput):
                ocr_method = output.method

        result = PipelineResult(
            status=status,
            receipt_data=receipt_data,
            receipt_id=receipt_id,
            error_message=error_message,
            stages=list(self._state.transitions),
            duration_ms=(end_time - time.time()) * 1000,  # Will be updated
            retry_count=retry_count,
            ocr_method=ocr_method,
            was_duplicate=was_duplicate,
            preserved_partial=self._state.validation_result.preserve_partial if self._state.validation_result else False
        )

        self._last_result = result
        return result

    # Assertion helpers

    def assert_persisted_once(self) -> None:
        """Assert exactly one receipt was persisted."""
        count = self.repository.get_save_count()
        if count != 1:
            raise AssertionError(f"Expected 1 save, got {count}")

    def assert_not_persisted(self) -> None:
        """Assert no receipts were persisted."""
        count = self.repository.get_save_count()
        if count != 0:
            raise AssertionError(f"Expected 0 saves, got {count}")

    def assert_retry_count(self, expected: int) -> None:
        """Assert specific retry count was used."""
        actual = self._last_result.retry_count if self._last_result else 0
        if actual != expected:
            raise AssertionError(f"Expected {expected} retries, got {actual}")

    def assert_stage_sequence(self, expected_stages: List[str]) -> None:
        """Assert pipeline executed stages in expected order.

        Args:
            expected_stages: List of stage names, e.g., ["OCR", "PARSING", "VALIDATION"]
        """
        actual_stages = [
            t.stage.name for t in self._state.transitions
        ]

        if actual_stages != expected_stages:
            raise AssertionError(
                f"Expected stages {expected_stages}, got {actual_stages}"
            )

    def assert_stage_executed(self, stage_name: str) -> None:
        """Assert a specific stage was executed."""
        executed = [t.stage.name for t in self._state.transitions]
        if stage_name not in executed:
            raise AssertionError(f"Stage {stage_name} was not executed. Stages: {executed}")

    def assert_ocr_used_fallback(self) -> None:
        """Assert OCR used Vision API fallback."""
        if not self.ocr.used_fallback():
            raise AssertionError("Expected OCR fallback, but primary OCR was used")

    def assert_duplicate_detected(self) -> None:
        """Assert duplicate receipt was detected."""
        if not self._last_result or not self._last_result.was_duplicate:
            raise AssertionError("Expected duplicate detection, but none occurred")

    def assert_validation_failed_on(self, field: str) -> None:
        """Assert validation failed on specific field."""
        from tests.harness.fakes.fake_validation_service import ValidationField
        field_enum = ValidationField(field)
        if not self.validator.did_field_fail(field_enum):
            raise AssertionError(f"Expected validation to fail on {field}, but it passed")

    def get_stage_timing(self, stage_name: str) -> Optional[float]:
        """Get duration in ms for a specific stage."""
        for transition in self._state.transitions:
            if transition.stage.name == stage_name and transition.exited_at:
                return (transition.exited_at - transition.entered_at).total_seconds() * 1000
        return None

    def get_full_report(self) -> Dict[str, Any]:
        """Get complete execution report for debugging."""
        return {
            "status": self._last_result.status.value if self._last_result else None,
            "stages": [
                {
                    "name": t.stage.name,
                    "duration_ms": self.get_stage_timing(t.stage.name),
                    "had_error": t.exception is not None,
                }
                for t in self._state.transitions
            ],
            "retry_count": self._last_result.retry_count if self._last_result else 0,
            "ocr_method": self._last_result.ocr_method if self._last_result else None,
            "repository_saves": self.repository.get_save_count(),
            "validation_failed_fields": self.validator.get_failed_fields(),
        }
