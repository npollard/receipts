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
class TraceEntry:
    """Single entry in execution trace.

    Captures stage execution for deterministic replay and debugging.
    """
    stage: str  # OCR, PARSE, VALIDATE, RETRY, PERSIST
    success: bool  # True if stage completed without error
    attempt: int  # Attempt number (1-based)
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None  # Additional context


@dataclass
class ExecutionTrace:
    """Ordered execution trace of pipeline stages.

    Provides deterministic record of exactly what happened during execution.
    """
    entries: List[TraceEntry] = field(default_factory=list)

    def add(self, stage: str, success: bool, attempt: int = 1, details: Optional[Dict] = None) -> None:
        """Add entry to trace."""
        self.entries.append(TraceEntry(
            stage=stage,
            success=success,
            attempt=attempt,
            timestamp=datetime.now(),
            details=details
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
    parsed_data: Optional[Dict[str, Any]] = None
    validation_result: Optional[ValidationResult] = None
    saved_receipt: Any = None
    current_stage: Optional[PipelineStage] = None
    transitions: List[StageTransition] = field(default_factory=list)
    trace: ExecutionTrace = field(default_factory=ExecutionTrace)


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
                   exception: Optional[Exception] = None, attempt: int = 1) -> None:
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
            PipelineStage.PARSING: "PARSE",
            PipelineStage.VALIDATION: "VALIDATE",
            PipelineStage.RETRY: "RETRY",
            PipelineStage.PERSISTENCE: "PERSIST",
        }
        stage_name = stage_name_map.get(stage, stage.name)
        success = exception is None

        self._state.trace.add(
            stage=stage_name,
            success=success,
            attempt=attempt,
            details={"output": output_data} if output_data else None
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
