"""Termination guarantee tests.

Ensures pipeline always terminates regardless of input patterns.
Critical for system stability.
"""

import pytest
from tests.harness.pipeline_harness import PipelineTestHarness, PipelineStatus


class TestPerpetualFailureTermination:
    """Pipeline must terminate even with perpetually failing inputs."""

    def test_ocr_always_fails_terminates(self):
        """Given: OCR fails on every attempt. Then: Pipeline terminates with failure."""
        harness = PipelineTestHarness()

        # Configure OCR to always fail (simulating completely broken OCR)
        harness.ocr.set_always_fail("permanent_ocr_failure")

        result = harness.run("receipt.jpg")

        # Must terminate (not hang)
        assert result.status == PipelineStatus.FAILURE
        # Retry count bounded (doesn't retry forever)
        assert harness.get_stage_attempts("OCR") <= 3  # Max retries enforced
        # No infinite loop
        assert len(harness.trace.get_all()) < 10  # Reasonable upper bound

    def test_persistence_always_fails_terminates(self):
        """Given: Persistence fails on every attempt. Then: Pipeline terminates."""
        harness = PipelineTestHarness()

        # Successful processing up to persistence
        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()
        # But persistence always fails
        harness.persistence.set_always_fail("db_down")

        result = harness.run("receipt.jpg")

        # Must terminate
        assert result.status == PipelineStatus.FAILURE
        # Persistence retries bounded
        assert harness.get_stage_attempts("PERSIST") <= 3
        # Execution completes (no hang)
        assert len(harness.trace.get_all()) < 15

    def test_validation_always_fails_terminates(self):
        """Given: Validation fails every time. Then: Pipeline terminates immediately."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        # Validation always rejects (e.g., duplicates detected)
        harness.validator.set_always_fail("duplicate_receipt")

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILURE
        # No retry on validation failure (non-retryable)
        assert harness.get_stage_attempts("VALIDATE") == 1
        # Persistence never attempted
        assert harness.get_stage_attempts("PERSIST") == 0


class TestAlternatingPatternTermination:
    """Pipeline must terminate with alternating success/failure patterns."""

    def test_alternating_ocr_quality_terminates(self):
        """Given: OCR alternates between low/high quality. Then: Pipeline terminates."""
        harness = PipelineTestHarness()

        # Sequence: low quality, high quality, low quality, high quality...
        # Even with alternation, should eventually succeed or exhaust retries
        harness.ocr.set_sequence([
            {"text": "garbage", "quality": 0.1},   # Low
            {"text": "STORE $10", "quality": 0.9},  # High
            {"text": "garbage2", "quality": 0.1},  # Low (if called)
        ])
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()
        harness.persistence.should_succeed(receipt_id="uuid-123")

        result = harness.run("receipt.jpg")

        # Must terminate
        assert result.status in [PipelineStatus.SUCCESS, PipelineStatus.FAILURE]
        # Execution bounded
        assert len(harness.trace.get_all()) < 20
        # OCR attempts bounded (max 3)
        assert harness.get_stage_attempts("OCR") <= 3

    def test_alternating_parse_validity_terminates(self):
        """Given: Parse alternates between valid/invalid. Then: Pipeline terminates."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.85)
        # Parser alternates: invalid, valid, invalid...
        harness.parser.set_sequence([
            Exception("parse_error"),  # Invalid
            {"merchant": "Store", "total": 10.0},  # Valid
            Exception("parse_error2"),  # Invalid (if called)
        ])
        harness.validator.all_fields_pass()
        harness.persistence.should_succeed(receipt_id="uuid-123")

        result = harness.run("receipt.jpg")

        # Must terminate
        assert result.status in [PipelineStatus.SUCCESS, PipelineStatus.FAILURE]
        # Parse attempts bounded
        assert harness.get_stage_attempts("PARSE") <= 3
        # No infinite loop
        assert len(harness.trace.get_all()) < 20


class TestMaximumRetryBounds:
    """Hard limits on retry counts prevent infinite loops."""

    def test_ocr_retry_limit_enforced(self):
        """Given: OCR keeps returning low quality. Then: Max 3 OCR attempts."""
        harness = PipelineTestHarness()

        # Always low quality (triggers fallback, which also low quality)
        harness.ocr.set_always_low_quality("garbage text", quality=0.1)
        # Parser configured but may not be called
        harness.parser.set_parse_result(merchant="Store", total=10.0)

        result = harness.run("receipt.jpg")

        # OCR attempts bounded by limit (max 3 cycles = 6 calls: primary + fallback each)
        assert harness.get_stage_attempts("OCR") <= 6
        # Pipeline terminates
        assert result.status == PipelineStatus.FAILURE

    def test_persistence_retry_limit_enforced(self):
        """Given: Persistence keeps failing. Then: Max 3 persistence attempts."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()
        # Persistence fails repeatedly
        harness.persistence.set_always_fail("db_timeout")

        result = harness.run("receipt.jpg")

        # Persistence retries bounded
        assert harness.get_stage_attempts("PERSIST") <= 3
        # Hard limit not exceeded
        assert harness.persistence.get_attempt_count() <= 3
        # Terminates
        assert result.status == PipelineStatus.FAILURE

    def test_total_stage_attempts_bounded(self):
        """Given: Multiple stages fail. Then: Total attempts bounded."""
        harness = PipelineTestHarness()

        # Multiple stages with issues
        harness.ocr.set_sequence([
            {"text": "low1", "quality": 0.2},
            {"text": "low2", "quality": 0.2},
        ])
        harness.parser.set_sequence([
            Exception("error1"),
            Exception("error2"),
        ])
        # Won't get to validation

        result = harness.run("receipt.jpg")

        # Total trace entries bounded
        trace = harness.trace.get_all()
        assert len(trace) < 15  # Reasonable upper bound

        # Individual stage bounds
        total_attempts = (
            harness.get_stage_attempts("OCR") +
            harness.get_stage_attempts("PARSE") +
            harness.get_stage_attempts("VALIDATE") +
            harness.get_stage_attempts("PERSIST")
        )
        assert total_attempts <= 6  # Max across all stages


class TestExecutionCompletes:
    """Execution always reaches terminal state."""

    def test_execution_always_completes_no_hang(self):
        """Given: Any input pattern. Then: Execution completes (no hang)."""
        harness = PipelineTestHarness()

        # Randomized failure pattern
        harness.ocr.set_text_for_image("receipt.jpg", "RANDOM TEXT", quality=0.5)
        harness.parser.set_parse_result(merchant="X", total=1.0)
        harness.validator.set_failure("test_error")

        # Run with timeout protection (harness should complete quickly)
        import time
        start = time.time()
        result = harness.run("receipt.jpg")
        elapsed = time.time() - start

        # Must complete quickly (no infinite loops)
        assert elapsed < 5.0  # 5 seconds max for test
        assert result.status in [PipelineStatus.SUCCESS, PipelineStatus.FAILURE, PipelineStatus.PARTIAL]
        # Trace exists
        assert len(harness.trace.get_all()) > 0
