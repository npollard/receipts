"""Execution trace tests using PipelineTestHarness.

Verifies trace captures exact execution sequences for:
- Happy path flows
- Retry scenarios
- Failure scenarios
- Fallback scenarios
"""

import pytest
from tests.harness.pipeline_harness import PipelineTestHarness, PipelineStatus
from tests.harness.fakes import OCROutput
from tests.harness.fakes.fake_receipt_parser import ParserOutput
from core.exceptions import OCRError


class TestHappyPathTrace:
    """Trace verification for successful pipeline execution."""

    def test_happy_path_trace(self):
        """Given: Normal receipt. When: Processed. Then: Trace shows OCR→PARSE→VALIDATE→PERSIST."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $25.00", quality=0.90)
        harness.parser.set_parse_result(merchant="Store", total=25.00)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS

        # Verify exact trace sequence
        harness.assert_trace_stage_sequence(["OCR", "PARSE", "VALIDATE", "PERSIST"])

        # Verify each stage executed exactly once
        harness.assert_trace_stage_count("OCR", 1)
        harness.assert_trace_stage_count("PARSE", 1)
        harness.assert_trace_stage_count("VALIDATE", 1)
        harness.assert_trace_stage_count("PERSIST", 1)

        # Verify no failures
        assert len(harness.trace.get_failed_stages()) == 0
        assert len(harness.trace.get_successful_stages()) == 4

    def test_happy_path_detailed_trace_entries(self):
        """Verify trace entries contain correct success/attempt info."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $30.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=30.00)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")

        # Check each entry
        entries = harness.trace.get_all()
        assert len(entries) == 4

        for entry in entries:
            assert entry.success is True
            assert entry.attempt == 1

        # Verify stages in order
        assert entries[0].stage == "OCR"
        assert entries[1].stage == "PARSE"
        assert entries[2].stage == "VALIDATE"
        assert entries[3].stage == "PERSIST"


class TestRetryTrace:
    """Trace verification for retry scenarios."""

    def test_retry_then_success_trace(self):
        """Given: Parser fails once then succeeds. When: Processed. Then: Trace shows single PARSE (retry is internal)."""
        harness = PipelineTestHarness(use_fake_retry=True)

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $40.00", quality=0.85)

        # Parser: fail, then succeed (retry handled internally)
        harness.parser.set_sequence([
            ValueError("Parse error"),
            ParserOutput(merchant="Store", total=40.00),
        ])

        harness.retry.set_succeed_on_attempt(2)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS

        # Trace shows single PARSE stage (retry happens within the stage)
        harness.assert_trace_stage_sequence(["OCR", "PARSE", "VALIDATE", "PERSIST"])
        harness.assert_trace_stage_count("PARSE", 1)

        # But retry count shows the internal retries
        assert result.retry_count == 1  # 1 retry = 2 attempts total

    def test_multiple_retries_trace(self):
        """Given: OCR fails then succeeds. When: Processed. Then: Trace shows multiple OCR entries."""
        harness = PipelineTestHarness()

        # OCR: fail twice, then succeed
        harness.ocr.set_sequence([
            OCRError("Error 1"),
            OCRError("Error 2"),
            OCROutput(text="SUCCESS", quality_score=0.88),
        ])

        harness.parser.set_parse_result(merchant="Store", total=50.00)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS

        # Trace shows 3 OCR attempts
        harness.assert_trace_stage_count("OCR", 3)

        # Other stages normal
        harness.assert_trace_stage_count("PARSE", 1)


class TestFailureTrace:
    """Trace verification for failure scenarios."""

    def test_full_failure_trace(self):
        """Given: OCR fails completely. When: Processed. Then: Trace shows OCR failure only."""
        harness = PipelineTestHarness()

        harness.ocr.set_should_fail(OCRError("Image unreadable"))

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED

        # Trace should only contain OCR (failed)
        harness.assert_trace_stage_sequence(["OCR", "OCR", "OCR"])  # 3 retry attempts
        harness.assert_trace_stage_count("OCR", 3)

        # All OCR attempts failed
        ocr_entries = [e for e in harness.trace.get_all() if e.stage == "OCR"]
        assert all(e.success is False for e in ocr_entries)

        # No other stages executed
        assert harness.trace.get_stage_attempts("PARSE") == 0
        assert harness.trace.get_stage_attempts("VALIDATE") == 0
        assert harness.trace.get_stage_attempts("PERSIST") == 0

    def test_validation_failure_trace(self):
        """Given: Validation fails. When: Processed. Then: Trace shows failure at validation."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $10.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=10.00)

        # Validation fails, no partial save
        harness.validator.all_fields_fail("Critical error")
        harness.validator.set_preserve_partial(False)

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.FAILED

        # Trace: OCR→PARSE→VALIDATE (no PERSIST)
        harness.assert_trace_stage_sequence(["OCR", "PARSE", "VALIDATE"])

        # Validation failed
        validate_entries = [e for e in harness.trace.get_all() if e.stage == "VALIDATE"]
        assert len(validate_entries) == 1
        assert validate_entries[0].success is True  # Stage completed, but validation returned invalid


class TestFallbackTrace:
    """Trace verification for OCR fallback scenarios."""

    def test_fallback_path_trace(self):
        """Given: Low quality OCR triggers fallback. When: Processed. Then: Trace shows both OCR attempts."""
        harness = PipelineTestHarness()

        # Low quality primary OCR
        harness.ocr.set_text_for_image("receipt.jpg", "BLURRY", quality=0.15)
        # High quality fallback
        harness.ocr.set_fallback_output("receipt.jpg", "CLEAR STORE $60.00", quality=0.90)

        harness.parser.set_parse_result(merchant="Store", total=60.00)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS

        # Trace should show OCR twice (primary + fallback)
        harness.assert_trace_stage_sequence(["OCR", "OCR", "PARSE", "VALIDATE", "PERSIST"])
        harness.assert_trace_stage_count("OCR", 2)

        # Both OCR entries succeeded
        ocr_entries = [e for e in harness.trace.get_all() if e.stage == "OCR"]
        assert all(e.success for e in ocr_entries)

    def test_fallback_failure_then_success_trace(self):
        """Given: OCR fails, then fallback succeeds. When: Processed. Then: Trace shows retry pattern."""
        harness = PipelineTestHarness()

        # First OCR attempt fails, fallback succeeds
        harness.ocr.set_sequence([
            OCRError("Primary failed"),
            OCROutput(text="FALLBACK SUCCESS", quality_score=0.88),
        ])

        harness.parser.set_parse_result(merchant="Store", total=70.00)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS

        # Should have 2 OCR attempts (one failed, one succeeded)
        harness.assert_trace_stage_count("OCR", 2)

        ocr_entries = [e for e in harness.trace.get_all() if e.stage == "OCR"]
        assert ocr_entries[0].success is False
        assert ocr_entries[1].success is True


class TestTraceDebugging:
    """Tests demonstrating clear debugging output on failure."""

    def test_trace_shows_detailed_failure_info(self):
        """Verify trace provides helpful debugging info when assertions fail."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $20.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=20.00)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")

        # Wrong sequence should produce helpful error
        try:
            harness.assert_trace_stage_sequence(["OCR", "PARSE", "PERSIST"])  # Missing VALIDATE
            pytest.fail("Should have raised AssertionError")
        except AssertionError as e:
            error_msg = str(e)
            assert "Expected trace sequence" in error_msg
            assert "got" in error_msg
            assert "Full trace:" in error_msg
            assert "OCR" in error_msg
            assert "PARSE" in error_msg
            assert "VALIDATE" in error_msg
            assert "PERSIST" in error_msg

    def test_retry_sequence_debugging_output(self):
        """Verify retry assertion provides clear debugging info."""
        harness = PipelineTestHarness()

        # OCR retry creates multiple OCR entries in trace
        harness.ocr.set_sequence([
            OCRError("Error 1"),
            OCRError("Error 2"),
            OCROutput(text="SUCCESS", quality_score=0.88),
        ])

        harness.parser.set_parse_result(merchant="Store", total=80.00)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")

        # Wrong attempt count should show helpful info
        try:
            harness.assert_trace_retry_sequence("OCR", 2)  # Actually 3
            pytest.fail("Should have raised AssertionError")
        except AssertionError as e:
            error_msg = str(e)
            assert "Expected 2 attempts" in error_msg
            assert "got 3" in error_msg
            assert "attempt=" in error_msg
            assert "success=" in error_msg

    def test_stage_count_debugging_output(self):
        """Verify stage count assertion provides helpful debugging info."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $15.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=15.00)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")

        # Wrong count should show all stages
        try:
            harness.assert_trace_stage_count("OCR", 2)  # Actually 1
            pytest.fail("Should have raised AssertionError")
        except AssertionError as e:
            error_msg = str(e)
            assert "Expected 2 occurrences of 'OCR'" in error_msg
            assert "got 1" in error_msg
            assert "Full trace stages:" in error_msg


class TestTraceExactMatching:
    """Tests for exact trace matching with success/attempt details."""

    def test_trace_matches_expected_exact(self):
        """Given: Happy path. When: Trace checked. Then: Exact match including success/attempt."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $35.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=35.00)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")

        # Exact match with (stage, success, attempt) tuples
        expected = [
            ("OCR", True, 1),
            ("PARSE", True, 1),
            ("VALIDATE", True, 1),
            ("PERSIST", True, 1),
        ]
        harness.assert_trace_matches_expected(expected, strict=True)

    def test_trace_matches_expected_with_retry(self):
        """Given: OCR retry scenario. When: Trace checked. Then: Shows OCR failure then success."""
        harness = PipelineTestHarness()

        harness.ocr.set_sequence([
            OCRError("Primary failed"),
            OCROutput(text="SUCCESS", quality_score=0.88),
        ])

        harness.parser.set_parse_result(merchant="Store", total=45.00)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")

        # Should show OCR failed then succeeded (both attempt=1 as separate stage calls)
        expected = [
            ("OCR", False, 1),
            ("OCR", True, 1),
            ("PARSE", True, 1),
            ("VALIDATE", True, 1),
            ("PERSIST", True, 1),
        ]
        harness.assert_trace_matches_expected(expected, strict=True)
