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
    PARSE = auto()
    VALIDATE = auto()
    RETRY = auto()
    PERSIST = auto()


class PipelineStatus(Enum):
    """Final status of pipeline execution."""
    SUCCESS = "success"
    FAILED = "failed"
    FAILURE = "failed"  # Alias for FAILED (test compatibility)
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
class TraceEntry:
    """Single entry in execution trace.

    Captures stage execution for deterministic replay and debugging.
    """
    stage: str  # OCR, PARSE, VALIDATE, RETRY, PERSIST
    success: bool  # True if stage completed without error
    attempt: int  # Attempt number (1-based)
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None  # Additional context (status, retry info, terminal flag)


@dataclass
class ExecutionTrace:
    """Ordered execution trace of pipeline stages.

    Provides deterministic record of exactly what happened during execution.
    """
    entries: List[TraceEntry] = field(default_factory=list)

    def add(self, stage: str, success: bool, attempt: int = 1, data: Optional[Dict] = None) -> None:
        """Add entry to trace."""
        self.entries.append(TraceEntry(
            stage=stage,
            success=success,
            attempt=attempt,
            timestamp=datetime.now(),
            data=data
        ))

    def get_all(self) -> List[TraceEntry]:
        """Get all trace entries in order."""
        return list(self.entries)

    def get_stage_attempts(self, stage: str) -> int:
        """Count attempts for a specific stage."""
        return sum(1 for e in self.entries if e.stage == stage)

    def get_failed_stages(self) -> List[str]:
        """Get names of stages that failed."""
        return [e.stage for e in self.entries if not e.success]

    def get_successful_stages(self) -> List[str]:
        """Get names of stages that succeeded."""
        return [e.stage for e in self.entries if e.success]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


@dataclass
class PipelineState:
    """Mutable state tracked during pipeline execution."""
    image_path: str = ""
    ocr_text: str = ""
    ocr_quality: float = 0.0
    ocr_method: str = ""  # "easyocr" or "vision"
    ocr_quality_exhausted: bool = False  # True if all OCR attempts had low quality
    parsed_data: Optional[Dict[str, Any]] = None
    validation_result: Optional[ValidationResult] = None
    saved_receipt: Any = None
    current_stage: Optional[PipelineStage] = None
    transitions: List[StageTransition] = field(default_factory=list)
    trace: ExecutionTrace = field(default_factory=ExecutionTrace)
    retry_count: int = 0  # Number of pipeline retry attempts (attempts - 1)


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
        >>> harness.assert_stage_sequence(["OCR", "PARSE", "VALIDATE", "PERSIST"])
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
        self.persistence = self.repository  # Alias for test convenience

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

    @property
    def trace(self) -> ExecutionTrace:
        """Get execution trace for the last pipeline run."""
        return self._state.trace

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

    def should_succeed(self, receipt_id: str) -> "PipelineTestHarness":
        """Configure persistence to succeed with given receipt ID."""
        self.persistence.should_succeed(receipt_id)
        return self

    def set_always_fail(self, error: str) -> "PipelineTestHarness":
        """Configure persistence to always fail with given error."""
        self.persistence.set_always_fail(error)
        return self

    def get_stage_attempts(self, stage: str) -> int:
        """Count attempts for a specific stage.

        Args:
            stage: Stage name (e.g., "OCR", "PARSE", "VALIDATE", "PERSIST")

        Returns:
            Number of times the stage was attempted
        """
        return self._state.trace.get_stage_attempts(stage)

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
                   exception: Optional[Exception] = None, attempt: int = 1,
                   data: Optional[Dict[str, Any]] = None) -> None:
        """Complete tracking of current stage and add to trace."""
        # Update stage transition
        for transition in reversed(self._state.transitions):
            if transition.stage == stage and transition.exited_at is None:
                transition.exited_at = datetime.now()
                transition.output_data = output_data
                transition.exception = exception
                break
        self._state.current_stage = None

        # Add to execution trace
        stage_name_map = {
            PipelineStage.OCR: "OCR",
            PipelineStage.PARSE: "PARSE",
            PipelineStage.VALIDATE: "VALIDATE",
            PipelineStage.RETRY: "RETRY",
            PipelineStage.PERSIST: "PERSIST",
        }
        stage_name = stage_name_map.get(stage, stage.name)
        success = exception is None

        # Build trace data merging output and provided data
        trace_data: Dict[str, Any] = {}
        if output_data is not None:
            trace_data["output"] = output_data
        if data is not None:
            trace_data.update(data)

        # Always include status for invariants
        if "status" not in trace_data:
            trace_data["status"] = "success" if success else "fail"

        self._state.trace.add(
            stage=stage_name,
            success=success,
            attempt=attempt,
            data=trace_data
        )

    def run(self, image_path: str) -> PipelineResult:
        """Execute the full pipeline on an image.

        Args:
            image_path: Path to the receipt image

        Returns:
            PipelineResult with full execution details
        """
        start_time = time.time()
        self._state = PipelineState(image_path=image_path)

        # Reset retry service history for fresh tracking
        if self._retry_is_fake:
            self.retry.reset_attempts()

        MAX_RETRIES = 3
        retry_count = 0

        # Stage 1: OCR (handles its own internal retries with fallback)
        ocr_result = self._run_ocr(image_path, max_attempts=3)
        if not ocr_result:
            return self._build_result(
                PipelineStatus.FAILED,
                error_message="OCR failed after all attempts"
            )

        # Stage 2: Check for duplicate by image hash
        existing = self._check_idempotency(image_path)
        if existing:
            # Handle both object.id and dict["id"] formats
            if isinstance(existing, dict):
                existing_id = existing.get('id')
            else:
                existing_id = getattr(existing, 'id', None)
            return self._build_result(
                PipelineStatus.DUPLICATE,
                receipt_id=existing_id,
                was_duplicate=True
            )

        # Stage 3: Parse (handles its own internal retries)
        parse_result = self._run_parsing(self._state.ocr_text, max_attempts=3)
        if not parse_result:
            self._state.retry_count = retry_count
            return self._build_result(PipelineStatus.FAILED, "Parsing failed")

        # Stage 4: Validate with stage-level retry
        validation_result = None
        for attempt in range(MAX_RETRIES):
            validation_result = self._run_validation(self._state.parsed_data, max_attempts=1)
            if not validation_result:
                self._state.retry_count = retry_count
                return self._build_result(PipelineStatus.FAILED, "Validation failed")

            if validation_result.is_valid:
                break  # Success - exit retry loop

            # Check if retryable
            if not getattr(validation_result, 'retryable', False):
                # Non-retryable - check if we should preserve partial
                if getattr(validation_result, 'preserve_partial', False):
                    # Will be handled in post-stage resolution
                    break
                # Otherwise return FAILURE immediately
                self._state.retry_count = retry_count
                return self._build_result(
                    PipelineStatus.FAILED,
                    error_message="Validation failed: " + "; ".join(validation_result.errors)
                )

            # Retryable error - check what stage to retry based on error
            if attempt < MAX_RETRIES - 1:
                retry_count += 1

                # Determine which stage to retry based on validation error
                errors = getattr(validation_result, 'errors', [])
                error_str = " ".join(errors).lower()

                # Check for OCR-related errors
                if any(e in error_str for e in ['ocr', 'text', 'image', 'quality', 'unreadable']):
                    # Retry OCR
                    self._state.ocr_text = ""
                    self._state.ocr_quality = 0.0
                    ocr_result = self._run_ocr(image_path, max_attempts=3)
                    if not ocr_result:
                        self._state.retry_count = retry_count
                        return self._build_result(
                            PipelineStatus.FAILED,
                            error_message="OCR failed on retry"
                        )
                    # Re-parse after successful OCR
                    parse_result = self._run_parsing(self._state.ocr_text, max_attempts=3)
                    if not parse_result:
                        self._state.retry_count = retry_count
                        return self._build_result(PipelineStatus.FAILED, "Parsing failed on retry")

                # Check for parsing-related errors
                elif any(e in error_str for e in ['parse', 'merchant', 'total', 'date', 'amount', 'malformed']):
                    # Retry parsing only
                    parse_result = self._run_parsing(self._state.ocr_text, max_attempts=3)
                    if not parse_result:
                        self._state.retry_count = retry_count
                        return self._build_result(PipelineStatus.FAILED, "Parsing failed on retry")

                # Otherwise, retry validation itself (data might be acceptable now)
                continue

            else:
                # No more attempts - break to post-stage resolution
                break

        else:
            # Loop completed without break (all retries exhausted)
            if not validation_result or not validation_result.is_valid:
                if getattr(validation_result, 'preserve_partial', False):
                    pass  # Will be handled in post-stage resolution
                else:
                    self._state.retry_count = retry_count
                    return self._build_result(
                        PipelineStatus.FAILED,
                        error_message="Validation failed after all retry attempts"
                    )

        # Store final retry count
        self._state.retry_count = retry_count

        # POST-STAGE RESOLUTION: Three outcomes based on validation result
        # Get final preserve_partial flag directly from validation_result
        final_preserve_partial = (
            getattr(validation_result, 'preserve_partial', False)
            if validation_result else False
        )

        # Outcome 1: SUCCESS - validation passed
        if validation_result and validation_result.is_valid:
            receipt = self._run_persistence(self._state.parsed_data, image_path)
            if not receipt:
                return self._build_result(PipelineStatus.FAILED, "Persistence failed")
            # Handle both object.id and dict["id"] formats
            if isinstance(receipt, dict):
                receipt_id = receipt.get('id')
            else:
                receipt_id = getattr(receipt, 'id', None)
            return self._build_result(
                PipelineStatus.SUCCESS,
                receipt_id=receipt_id,
                receipt_data=self._state.parsed_data
            )

        # Outcome 2: PARTIAL - validation failed but preserve_partial is True
        if final_preserve_partial and self._state.parsed_data:
            receipt = self._run_persistence(self._state.parsed_data, image_path)
            if not receipt:
                return self._build_result(PipelineStatus.FAILED, "Persistence failed")
            # Handle both object.id and dict["id"] formats
            if isinstance(receipt, dict):
                receipt_id = receipt.get('id')
            else:
                receipt_id = getattr(receipt, 'id', None)
            return self._build_result(
                PipelineStatus.PARTIAL,
                receipt_id=receipt_id,
                receipt_data=self._state.parsed_data
            )

        # Outcome 3: FAILURE - all other cases
        return self._build_result(PipelineStatus.FAILED, "Pipeline exhausted all attempts")

    def _run_ocr(self, image_path: str, max_attempts: int = 3) -> Optional[str]:
        """Execute OCR stage with retry support and fallback on low quality.

        Args:
            image_path: Path to image
            max_attempts: Maximum OCR attempts (default 3)

        Returns:
            OCR text if successful, None if all failed
        """
        QUALITY_THRESHOLD = 0.5

        for attempt in range(max_attempts):
            # Try primary OCR (EasyOCR)
            self._start_stage(PipelineStage.OCR, {"image_path": image_path, "method": "easyocr", "attempt": attempt + 1})

            try:
                text = self.ocr.extract_text(image_path, use_vision_fallback=False)

                # Get quality score
                try:
                    quality = self.ocr.score_ocr_quality(text)
                except Exception:
                    quality = 0.5

                # Check if quality is acceptable
                if quality >= QUALITY_THRESHOLD:
                    # Good quality - record success and return
                    self._state.ocr_text = text
                    self._state.ocr_quality = quality
                    self._state.ocr_method = "easyocr"

                    self._end_stage(PipelineStage.OCR, {
                        "text": text,
                        "quality": quality,
                        "success": True,
                        "method": "easyocr"
                    })
                    return text

                # Low quality - try fallback (Vision API)
                # Primary succeeded but quality is low - mark success=True with low_quality flag
                self._end_stage(PipelineStage.OCR, {
                    "text": text,
                    "quality": quality,
                    "success": True,
                    "low_quality": True,
                    "method": "easyocr"
                })

                # Attempt fallback
                self._start_stage(PipelineStage.OCR, {"image_path": image_path, "method": "vision", "fallback": True, "attempt": attempt + 1})

                try:
                    fallback_text = self.ocr.extract_text(image_path, use_vision_fallback=True)

                    # Get fallback quality
                    try:
                        fallback_quality = self.ocr.score_ocr_quality(fallback_text)
                    except Exception:
                        fallback_quality = 0.5

                    # Check fallback quality
                    if fallback_quality >= QUALITY_THRESHOLD:
                        # Fallback succeeded with good quality
                        self._state.ocr_text = fallback_text
                        self._state.ocr_quality = fallback_quality
                        self._state.ocr_method = "vision"

                        self._end_stage(PipelineStage.OCR, {
                            "text": fallback_text,
                            "quality": fallback_quality,
                            "success": True,
                            "method": "vision",
                            "fallback": True
                        })
                        return fallback_text

                    # Fallback also low quality - record and continue to next attempt
                    # Fallback succeeded but quality is low - mark success=True with low_quality flag
                    self._end_stage(PipelineStage.OCR, {
                        "text": fallback_text,
                        "quality": fallback_quality,
                        "success": True,
                        "low_quality": True,
                        "method": "vision",
                        "fallback": True
                    })

                except Exception as e:
                    # Fallback failed
                    self._end_stage(PipelineStage.OCR, {
                        "success": False,
                        "method": "vision",
                        "fallback": True,
                        "error": str(e)
                    }, exception=e)

                # Both primary and fallback failed on this attempt, continue to next
                continue

            except Exception as e:
                # Primary OCR failed
                self._end_stage(PipelineStage.OCR, {
                    "success": False,
                    "method": "easyocr"
                }, exception=e)
                # Continue to next attempt
                continue

        # All attempts exhausted
        self._state.ocr_quality_exhausted = True
        return None

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

    def _run_parsing(self, text: str, max_attempts: int = 3) -> bool:
        """Execute parsing stage with internal retry support.

        Args:
            text: OCR text to parse
            max_attempts: Maximum parsing attempts (default 3)

        Returns:
            True if parsing succeeded, False if all failed
        """
        # Start PARSE stage once (internal retries are not traced)
        self._start_stage(PipelineStage.PARSE, {"text": text, "max_attempts": max_attempts})

        # Use FakeRetryService when available for proper retry tracking
        if self._retry_is_fake:
            try:
                def parse_with_retry():
                    response = self.parser.parse_text(text)
                    if response.status != "success":
                        raise ValueError(f"Parse failed: {response.error}")
                    return response.data

                parsed_data = self.retry.execute_with_retry(parse_with_retry)
                self._state.parsed_data = parsed_data
                self._end_stage(PipelineStage.PARSE, parsed_data, data={"status": "success", "attempts": self.retry.get_attempt_count()})
                return True
            except Exception as e:
                self._end_stage(PipelineStage.PARSE, exception=e, data={"status": "fail", "attempts": self.retry.get_attempt_count()})
                return False

        # Fallback to manual retry loop
        last_exception = None
        for attempt in range(max_attempts):
            try:
                response = self.parser.parse_text(text)
                if response.status == "success":
                    self._state.parsed_data = response.data
                    self._end_stage(PipelineStage.PARSE, response.data, data={"status": "success", "attempts": attempt + 1})
                    return True
                else:
                    last_exception = ValueError(f"Parse failed: {response.error}")
                    # Continue to next attempt
            except Exception as e:
                last_exception = e
                # Continue to next attempt

        # All attempts failed
        self._end_stage(PipelineStage.PARSE, exception=last_exception, data={"status": "fail", "attempts": max_attempts})
        return False

    def _run_validation(self, data: Dict[str, Any], max_attempts: int = 1) -> Optional[ValidationResult]:
        """Execute validation stage with optional retry for retryable failures.

        Args:
            data: Receipt data to validate
            max_attempts: Maximum validation attempts (default 1, only retry if retryable=True)

        Returns:
            ValidationResult if successful, None if failed
        """
        # Start VALIDATE stage once
        self._start_stage(PipelineStage.VALIDATE, {"data": data})

        last_exception = None
        for attempt in range(max_attempts):
            try:
                result = self.validator.validate_receipt(data)
                self._state.validation_result = result
                status_str = "pass" if result.is_valid else "fail"
                self._end_stage(PipelineStage.VALIDATE, result, data={"status": status_str, "attempts": attempt + 1})
                return result
            except Exception as e:
                last_exception = e
                # Continue to next attempt if retries remain

        # All attempts failed
        self._end_stage(PipelineStage.VALIDATE, exception=last_exception, data={"status": "fail", "attempts": max_attempts})
        return self._state.validation_result

    def _run_persistence(self, data: Dict[str, Any], image_path: str, max_attempts: int = 3) -> Optional[Any]:
        """Execute persistence stage with internal retry support.

        Args:
            data: Receipt data to persist
            image_path: Path to the receipt image
            max_attempts: Maximum persistence attempts (default 3)

        Returns:
            Saved receipt if successful, None if all failed
        """
        # Get image hash once (doesn't change between retries)
        image_hash = self._image_hashes.get(image_path, f"hash_{Path(image_path).stem}")

        # Preserve configured receipt_id across retries
        configured_id = getattr(self.repository, '_next_receipt_id', None)

        # Start PERSIST stage once (internal retries are not traced)
        self._start_stage(PipelineStage.PERSIST, {"data": data, "max_attempts": max_attempts})

        last_exception = None
        for attempt in range(max_attempts):
            # Re-configure the receipt_id if it was consumed by previous attempt
            current_id = getattr(self.repository, '_next_receipt_id', None)
            if configured_id is not None and current_id is None:
                self.repository._next_receipt_id = configured_id

            try:
                receipt = self.repository.save_receipt(
                    user_id=self._user_id,
                    image_path=image_path,
                    receipt_data=data,
                    image_hash=image_hash
                )

                self._state.saved_receipt = receipt
                self._end_stage(PipelineStage.PERSIST, receipt, data={"status": "success", "attempts": attempt + 1})
                return receipt

            except Exception as e:
                last_exception = e
                # Continue to next attempt

        # All attempts failed
        self._end_stage(PipelineStage.PERSIST, exception=last_exception, data={"status": "fail", "attempts": max_attempts})
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

        # Mark last trace entry as terminal for terminal states
        terminal_states = {PipelineStatus.FAILED, PipelineStatus.FAILURE, PipelineStatus.SUCCESS, PipelineStatus.PARTIAL, PipelineStatus.DUPLICATE}
        if status in terminal_states and self._state.trace.entries:
            last_entry = self._state.trace.entries[-1]
            if last_entry.data is None:
                last_entry.data = {}
            last_entry.data["terminal"] = True

        # Validate all stages use standardized naming
        valid_stages = {"OCR", "PARSE", "VALIDATE", "RETRY", "PERSIST"}
        for transition in self._state.transitions:
            stage_name = transition.stage.name
            assert stage_name in valid_stages, (
                f"Invalid stage name '{stage_name}'. "
                f"Must be one of: {valid_stages}"
            )

        # Get retry count from fake retry service when available, otherwise from state
        if self._retry_is_fake:
            retry_count = max(0, self.retry.get_attempt_count() - 1)
        else:
            retry_count = self._state.retry_count if self._state else 0

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
        # When using fake retry, check the retry service's attempt count
        if self._retry_is_fake:
            actual = max(0, self.retry.get_attempt_count() - 1)
        else:
            actual = self._last_result.retry_count if self._last_result else 0
        if actual != expected:
            raise AssertionError(f"Expected {expected} retries, got {actual}")

    def assert_stage_sequence(self, expected_stages: List[str]) -> None:
        """Assert pipeline executed stages in expected order.

        Collapses consecutive duplicates in actual trace to handle retries
        (e.g., ["OCR", "OCR", "PARSE"] → ["OCR", "PARSE"])

        Args:
            expected_stages: Normalized list of stage names, e.g., ["OCR", "PARSE", "VALIDATE"]
        """
        raw_stages = [t.stage.name for t in self._state.transitions]

        # Collapse consecutive duplicates
        normalized_stages = []
        for stage in raw_stages:
            if not normalized_stages or normalized_stages[-1] != stage:
                normalized_stages.append(stage)

        if normalized_stages != expected_stages:
            raise AssertionError(
                f"Expected stages {expected_stages}, got {normalized_stages} (raw: {raw_stages})"
            )

    def assert_stage_sequence_normalized(self, expected_stages: List[str]) -> None:
        """Assert pipeline executed stages in order, ignoring consecutive duplicates.

        Normalizes actual stages by removing consecutive duplicates
        to handle retry scenarios (e.g., ["OCR", "OCR", "PARSE"] → ["OCR", "PARSE"])

        Args:
            expected_stages: Normalized list of stage names, e.g., ["OCR", "PARSE", "VALIDATE"]
        """
        raw_stages = [t.stage.name for t in self._state.transitions]

        # Normalize: remove consecutive duplicates
        normalized_stages = []
        for stage in raw_stages:
            if not normalized_stages or normalized_stages[-1] != stage:
                normalized_stages.append(stage)

        if normalized_stages != expected_stages:
            raise AssertionError(
                f"Expected stages {expected_stages}, got {normalized_stages} (raw: {raw_stages})"
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

    # Trace-based assertions

    def assert_trace_stage_sequence(self, expected_sequence: List[str]) -> None:
        """Assert trace contains exact stage sequence.

        Args:
            expected_sequence: List of stage names in expected order,
                              e.g., ["OCR", "PARSE", "VALIDATE", "PERSIST"]

        Raises:
            AssertionError: If actual sequence doesn't match expected
        """
        actual = [entry.stage for entry in self.trace.get_all()]

        if actual != expected_sequence:
            raise AssertionError(
                f"Expected trace sequence {expected_sequence},\n"
                f"got {actual}\n\n"
                f"Full trace:\n" +
                "\n".join(f"  {i}: {e.stage} (success={e.success}, attempt={e.attempt})"
                         for i, e in enumerate(self.trace.get_all()))
            )

    def assert_trace_stage_count(self, stage: str, expected_count: int) -> None:
        """Assert trace contains exact number of stage occurrences.

        Args:
            stage: Stage name to count (e.g., "PARSE")
            expected_count: Expected number of occurrences

        Raises:
            AssertionError: If count doesn't match expected
        """
        actual = self.trace.get_stage_attempts(stage)

        if actual != expected_count:
            all_stages = [e.stage for e in self.trace.get_all()]
            raise AssertionError(
                f"Expected {expected_count} occurrences of '{stage}', got {actual}\n"
                f"Full trace stages: {all_stages}"
            )

    def assert_trace_retry_sequence(self, stage: str, expected_attempts: int) -> None:
        """Assert retry sequence for a specific stage.

        Verifies that a stage was retried the expected number of times,
        with failures followed by eventual success.

        Args:
            stage: Stage name that was retried (e.g., "PARSE")
            expected_attempts: Total expected attempts (failures + 1 success)

        Raises:
            AssertionError: If retry pattern doesn't match expected
        """
        entries = [e for e in self.trace.get_all() if e.stage == stage]

        if len(entries) != expected_attempts:
            raise AssertionError(
                f"Expected {expected_attempts} attempts for '{stage}', got {len(entries)}\n"
                f"Trace entries for '{stage}':\n" +
                "\n".join(f"  attempt={e.attempt}, success={e.success}"
                         for e in entries)
            )

        # Verify pattern: all but last should fail, last should succeed
        for i, entry in enumerate(entries):
            if i < len(entries) - 1:
                if entry.success:
                    raise AssertionError(
                        f"Expected attempt {i+1} of '{stage}' to fail (retry), "
                        f"but it succeeded\n"
                        f"Full sequence: " +
                        ", ".join(f"{e.attempt}:{'✓' if e.success else '✗'}"
                                 for e in entries)
                    )
            else:
                if not entry.success:
                    raise AssertionError(
                        f"Expected final attempt of '{stage}' to succeed, "
                        f"but it failed\n"
                        f"Full sequence: " +
                        ", ".join(f"{e.attempt}:{'✓' if e.success else '✗'}"
                                 for e in entries)
                    )

    def assert_trace_matches_expected(
        self,
        expected: List[tuple],
        strict: bool = True
    ) -> None:
        """Assert trace matches expected pattern.

        Args:
            expected: List of (stage, success, attempt) tuples
            strict: If True, lengths must match exactly; if False,
                   only checks that expected entries exist in order

        Raises:
            AssertionError: If trace doesn't match expected pattern
        """
        actual = [(e.stage, e.success, e.attempt) for e in self.trace.get_all()]

        if strict:
            if actual != expected:
                raise AssertionError(
                    f"Trace mismatch!\n\n"
                    f"Expected:\n" +
                    "\n".join(f"  {i}: stage={s}, success={su}, attempt={a}"
                             for i, (s, su, a) in enumerate(expected)) +
                    f"\n\nActual:\n" +
                    "\n".join(f"  {i}: stage={s}, success={su}, attempt={a}"
                             for i, (s, su, a) in enumerate(actual))
                )
        else:
            # Check that expected entries exist in order
            actual_idx = 0
            for exp_stage, exp_success, exp_attempt in expected:
                found = False
                while actual_idx < len(actual):
                    act_stage, act_success, act_attempt = actual[actual_idx]
                    actual_idx += 1
                    if (act_stage == exp_stage and
                        act_success == exp_success and
                        act_attempt == exp_attempt):
                        found = True
                        break
                if not found:
                    raise AssertionError(
                        f"Expected trace entry not found: "
                        f"stage={exp_stage}, success={exp_success}, attempt={exp_attempt}\n"
                        f"Full trace: {actual}"
                    )

    # Idempotency assertions

    def assert_no_duplicate_persistence(self) -> None:
        """Assert that duplicate receipts were not persisted (idempotency maintained).

        Verifies that save attempts for duplicate content did not result in additional writes.
        """
        metrics = self.repository.get_metrics()
        actual_writes = metrics.actual_writes
        save_attempts = metrics.save_attempts
        duplicates = metrics.duplicate_detections

        if actual_writes != save_attempts - duplicates:
            raise AssertionError(
                f"Duplicate persistence detected. "
                f"Expected {save_attempts - duplicates} actual writes, "
                f"got {actual_writes}. "
                f"Save attempts: {save_attempts}, "
                f"Duplicates detected: {duplicates}"
            )

    def assert_persist_count(self, expected: int) -> None:
        """Assert exactly N receipts were persisted (actual DB writes).

        This counts only successful non-duplicate writes, not save() calls.

        Args:
            expected: Expected number of persisted receipts
        """
        metrics = self.repository.get_metrics()
        actual = metrics.actual_writes

        if actual != expected:
            raise AssertionError(
                f"Expected {expected} persisted receipts, got {actual}. "
                f"Save attempts: {metrics.save_attempts}, "
                f"Duplicates: {metrics.duplicate_detections}"
            )

    def assert_unique_hashes(self, expected: int) -> None:
        """Assert repository contains exactly N unique content hashes.

        This verifies deduplication is working - multiple identical receipts
        should only store one hash.

        Args:
            expected: Expected number of unique content hashes
        """
        all_receipts = self.repository.get_all()
        unique_hashes = set()
        for receipt in all_receipts:
            hash_val = getattr(receipt, 'receipt_data_hash', None)
            if hash_val:
                unique_hashes.add(hash_val)

        actual = len(unique_hashes)
        if actual != expected:
            raise AssertionError(
                f"Expected {expected} unique content hashes, got {actual}. "
                f"Total receipts in store: {len(all_receipts)}, "
                f"Unique hashes: {sorted(unique_hashes)}"
            )

    # State exposure for debugging

    def get_persisted_records(self) -> List[Any]:
        """Get all persisted receipt records from repository.

        Returns:
            List of receipt DTOs currently in the repository
        """
        return self.repository.get_all()

    def get_content_hashes(self) -> List[str]:
        """Get all unique content hashes currently stored.

        Returns:
            List of unique data hash values
        """
        all_receipts = self.repository.get_all()
        hashes = set()
        for receipt in all_receipts:
            hash_val = getattr(receipt, 'receipt_data_hash', None)
            if hash_val:
                hashes.add(hash_val)
        return sorted(list(hashes))

    def get_image_hashes(self) -> List[str]:
        """Get all image hashes used during processing.

        Returns:
            List of image hash values
        """
        return sorted(list(self._image_hashes.values()))

    def get_repository_metrics(self) -> Any:
        """Get detailed repository operation metrics.

        Returns:
            RepositoryMetrics with save attempts, actual writes, duplicates, etc.
        """
        return self.repository.get_metrics()

    def get_full_report(self) -> Dict[str, Any]:
        """Get complete execution report for debugging."""
        metrics = self.repository.get_metrics()
        # Use fake retry service count when available
        if self._retry_is_fake and self._last_result:
            retry_count = max(0, self.retry.get_attempt_count() - 1)
        else:
            retry_count = self._last_result.retry_count if self._last_result else 0
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
            "retry_count": retry_count,
            "ocr_method": self._last_result.ocr_method if self._last_result else None,
            "repository_saves": self.repository.get_save_count(),
            "repository_metrics": {
                "save_attempts": metrics.save_attempts,
                "actual_writes": metrics.actual_writes,
                "duplicate_detections": metrics.duplicate_detections,
                "constraint_violations": metrics.constraint_violations,
            },
            "persisted_count": len(self.get_persisted_records()),
            "unique_hashes": len(self.get_content_hashes()),
            "validation_failed_fields": self.validator.get_failed_fields(),
        }
