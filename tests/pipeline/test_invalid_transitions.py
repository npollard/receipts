"""Invalid transition tests.

Ensures pipeline does NOT perform prohibited behaviors.
Negative tests for system correctness.
"""

import pytest
from tests.harness.pipeline_harness import PipelineTestHarness, PipelineStatus


class TestNoRetryOnSuccess:
    """Retry should not trigger on successful stages."""

    def test_validation_pass_no_retry(self):
        """Given: Validation passes. Then: No retry should occur."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()
        harness.persistence.should_succeed(receipt_id="uuid-123")

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        # Critical: Only 1 attempt per stage (no retries)
        assert harness.get_stage_attempts("OCR") == 1
        assert harness.get_stage_attempts("PARSE") == 1
        assert harness.get_stage_attempts("VALIDATE") == 1
        assert harness.get_stage_attempts("PERSIST") == 1
        # No retry stages in trace
        trace = [e.stage for e in harness.trace.get_all()]
        assert trace.count("OCR") == 1
        assert trace.count("PARSE") == 1

    def test_persistence_success_no_retry(self):
        """Given: Persistence succeeds. Then: No persistence retry should occur."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()
        harness.persistence.should_succeed(receipt_id="uuid-456")

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        # Exactly 1 persistence attempt
        assert harness.get_stage_attempts("PERSIST") == 1
        # No duplicate persistence calls
        assert harness.persistence.get_attempt_count() == 1


class TestNoParseOnOCRFailure:
    """Parse should not run when OCR fails completely."""

    def test_ocr_failure_no_parse_attempt(self):
        """Given: OCR fails with no output. Then: Parse should NOT execute."""
        harness = PipelineTestHarness()

        # Complete OCR failure - no text extracted
        harness.ocr.set_failure("receipt.jpg", "ocr_crash")
        # Configure parser but it should never be called
        harness.parser.set_parse_result(merchant="ShouldNotRun", total=999.0)

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILURE
        # OCR attempted (and failed)
        assert harness.get_stage_attempts("OCR") >= 1
        # Parse should NOT have been called
        assert harness.get_stage_attempts("PARSE") == 0
        # Verify parser was never invoked
        assert len(harness.parser.get_attempt_history()) == 0


class TestNoStagesAfterTerminal:
    """No stages should execute after terminal state."""

    def test_validation_fail_no_persistence(self):
        """Given: Validation fails terminally. Then: Persistence should NOT execute."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        # Validation fails - should be terminal
        from tests.harness.fakes.fake_validation_service import ValidationField
        harness.validator.all_fields_fail("Critical validation error")
        # Persistence configured but should never run
        harness.persistence.should_succeed(receipt_id="uuid-never")

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILURE
        # OCR and parse ran
        assert harness.get_stage_attempts("OCR") == 1
        assert harness.get_stage_attempts("PARSE") == 1
        # Validation ran and failed
        assert harness.get_stage_attempts("VALIDATE") == 1
        # Persistence should NOT have run
        assert harness.get_stage_attempts("PERSIST") == 0
        # Verify no persistence calls
        assert harness.persistence.get_attempt_count() == 0

    def test_terminal_failure_halts_immediately(self):
        """Given: Terminal failure occurs. Then: All stage execution stops."""
        harness = PipelineTestHarness()

        # OCR succeeds
        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.85)
        # Parse succeeds
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        # Validation fails terminally (e.g., duplicate)
        harness.validator.set_failure("duplicate_receipt")
        # No further stages should run

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILURE
        trace = harness.trace.get_all()
        stage_names = [e.stage for e in trace]

        # Find where validation failed
        validation_index = None
        for i, name in enumerate(stage_names):
            if name == "VALIDATE":
                validation_index = i
                break

        assert validation_index is not None
        # No stages after validation
        post_validation = stage_names[validation_index + 1:]
        # Only acceptable post-validation stage is persistence (if validation passed)
        # But since validation failed, there should be no persistence
        assert "PERSIST" not in post_validation, (
            f"Persistence executed after validation failure. "
            f"Full trace: {stage_names}"
        )


class TestNoPersistenceWithoutValidation:
    """Persistence should only occur after validation passes."""

    def test_persistence_only_after_validation_pass(self):
        """Persistence attempts should never precede validation success."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()
        harness.persistence.should_succeed(receipt_id="uuid-789")

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS
        trace = harness.trace.get_all()

        # Find validation and persistence indices
        validation_index = None
        persistence_index = None
        for i, entry in enumerate(trace):
            if entry.stage == "VALIDATE":
                validation_index = i
            elif entry.stage == "PERSIST":
                persistence_index = i

        # Both should exist
        assert validation_index is not None, "Validation not in trace"
        assert persistence_index is not None, "Persistence not in trace"
        # Persistence must come AFTER validation
        assert persistence_index > validation_index, (
            f"Persistence (index {persistence_index}) occurred before "
            f"validation (index {validation_index})"
        )
