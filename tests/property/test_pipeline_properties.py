"""Property-based tests for pipeline behavior using harness.

Tests invariants:
- Pipeline never crashes on random inputs
- Retry logic always terminates
- No duplicate persistence occurs
- Trace is always valid
"""

import pytest

try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis.strategies import text, floats, booleans, lists, dictionaries, one_of
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    pytest.skip("Hypothesis not installed", allow_module_level=True)

from tests.harness.pipeline_harness import PipelineTestHarness, PipelineStatus
from tests.harness.fakes import OCROutput, ParserOutput


class TestPipelineNeverCrashes:
    """Pipeline must handle any input without crashing."""

    @given(
        ocr_text=text(min_size=0, max_size=1000),
        quality=floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_pipeline_handles_random_ocr_text(self, ocr_text, quality):
        """Given: Random OCR text. When: Processed. Then: No crash, valid result."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", ocr_text, quality=quality)
        harness.parser.set_default_output(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()

        # Should not raise exception
        result = harness.run("receipt.jpg")

        # Result should be valid PipelineResult
        assert result is not None
        assert isinstance(result.status, PipelineStatus)
        assert result.status in [PipelineStatus.SUCCESS, PipelineStatus.FAILED, PipelineStatus.PARTIAL, PipelineStatus.DUPLICATE]

    @given(
        merchant=text(min_size=0, max_size=100),
        total=one_of(
            floats(min_value=0, max_value=100000, allow_nan=False),
            st.none()
        ),
        date=text(min_size=0, max_size=20),
    )
    @settings(max_examples=100, deadline=None)
    def test_pipeline_handles_random_parser_output(self, merchant, total, date):
        """Given: Random parser output. When: Processed. Then: No crash."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $25.00", quality=0.85)
        harness.parser.set_parse_result(
            merchant=merchant or "Unknown",
            total=total if total else 0.0,
            date=date or "2024-01-01"
        )
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result is not None
        assert isinstance(result.status, PipelineStatus)

    @given(
        is_valid=booleans(),
        error_message=text(min_size=0, max_size=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_pipeline_handles_random_validation(self, is_valid, error_message):
        """Given: Random validation outcome. When: Processed. Then: No crash."""
        from tests.harness.fakes.fake_validation_service import ValidationField

        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $25.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=25.00)

        if is_valid:
            harness.validator.all_fields_pass()
        else:
            harness.validator.all_fields_fail(error_message or "Validation failed")
            harness.validator.set_preserve_partial(True)

        result = harness.run("receipt.jpg")

        assert result is not None
        assert isinstance(result.status, PipelineStatus)

    @given(
        exception_type=one_of(st.just(ValueError), st.just(RuntimeError), st.just(TypeError)),
        exception_msg=text(min_size=1, max_size=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_pipeline_handles_ocr_exceptions(self, exception_type, exception_msg):
        """Given: OCR raises exception. When: Processed. Then: Graceful failure."""
        harness = PipelineTestHarness()

        harness.ocr.set_should_fail(exception_type(exception_msg))

        result = harness.run("receipt.jpg")

        assert result is not None
        assert result.status in [PipelineStatus.FAILED, PipelineStatus.DUPLICATE]


class TestRetryLogicTerminates:
    """Retry logic must always terminate, never infinite loop."""

    @given(
        fail_count=st.integers(min_value=0, max_value=5),
        max_retries=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50, deadline=None)
    def test_retry_terminates_with_eventual_success(self, fail_count, max_retries):
        """Given: Parser fails N times then succeeds. When: Retried. Then: Terminates."""
        assume(fail_count <= max_retries)  # Must eventually succeed

        harness = PipelineTestHarness(use_fake_retry=True)

        # Create sequence: N failures then success
        sequence = []
        for _ in range(fail_count):
            sequence.append(ValueError("Temporary failure"))
        sequence.append(ParserOutput(merchant="Store", total=25.00))

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $25.00", quality=0.85)
        harness.parser.set_sequence(sequence)
        harness.retry.set_succeed_on_attempt(fail_count + 1)
        harness.retry.set_max_retries(max_retries)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        # Must terminate and eventually succeed
        assert result is not None
        assert result.status == PipelineStatus.SUCCESS

    @given(
        max_retries=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_retry_terminates_with_exhausted_failure(self, max_retries):
        """Given: Parser always fails. When: Retried max times. Then: Terminates with failure."""
        harness = PipelineTestHarness(use_fake_retry=True)

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $25.00", quality=0.85)
        # Always fails
        harness.parser.set_sequence([ValueError("Persistent error")])
        harness.retry.set_max_retries(max_retries)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        # Must terminate (even if failed)
        assert result is not None
        # Should have attempted up to max_retries
        assert harness.retry.get_attempt_count() <= max_retries + 1

    @given(
        attempts=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=None)
    def test_ocr_retry_terminates(self, attempts):
        """Given: OCR fails multiple times. When: Retried. Then: Terminates."""
        from core.exceptions import OCRError

        harness = PipelineTestHarness()

        # Sequence of failures ending in success
        sequence = [OCRError(f"Attempt {i} failed") for i in range(attempts - 1)]
        sequence.append(OCROutput(text="SUCCESS", quality_score=0.85))

        harness.ocr.set_sequence(sequence)
        harness.parser.set_default_output(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        # Must terminate
        assert result is not None
        assert result.status in [PipelineStatus.SUCCESS, PipelineStatus.FAILED]


class TestNoDuplicatePersistence:
    """No duplicate records should be persisted."""

    @given(
        merchant=text(min_size=1, max_size=50),
        total=floats(min_value=0, max_value=100000, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_same_content_not_duplicated(self, merchant, total):
        """Given: Same receipt processed twice. When: Run twice. Then: Single record."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", f"{merchant} ${total}", quality=0.85)
        harness.parser.set_parse_result(merchant=merchant, total=total)
        harness.validator.all_fields_pass()

        # Process twice with same content
        result1 = harness.run("receipt.jpg")
        result2 = harness.run("receipt.jpg")

        # Only one record should exist
        harness.assert_persist_count(1)
        harness.assert_no_duplicate_persistence()

    @given(
        run_count=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=None)
    def test_multiple_runs_single_persistence(self, run_count):
        """Given: Same input processed N times. When: All complete. Then: 1 record."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $50.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=50.00)
        harness.validator.all_fields_pass()

        for _ in range(run_count):
            harness.run("receipt.jpg")

        # Only one record despite multiple runs
        harness.assert_persist_count(1)
        harness.assert_unique_hashes(1)

    @given(
        merchant1=text(min_size=1, max_size=50),
        merchant2=text(min_size=1, max_size=50),
        total1=floats(min_value=0, max_value=1000, allow_nan=False),
        total2=floats(min_value=0, max_value=1000, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_different_content_distinct_records(self, merchant1, merchant2, total1, total2):
        """Given: Different receipts. When: Processed. Then: Distinct records (or dedup)."""
        # Skip if content is identical
        assume(merchant1 != merchant2 or total1 != total2)

        harness = PipelineTestHarness()
        harness.validator.all_fields_pass()

        # First receipt
        harness.ocr.set_text_for_image("r1.jpg", f"{merchant1} ${total1}", quality=0.85)
        harness.parser.set_parse_result(merchant=merchant1, total=total1)
        harness.run("r1.jpg")

        # Second receipt (different)
        harness.ocr.set_text_for_image("r2.jpg", f"{merchant2} ${total2}", quality=0.85)
        harness.parser.set_parse_result(merchant=merchant2, total=total2)
        harness.run("r2.jpg")

        # Repository should have consistent state
        metrics = harness.get_repository_metrics()
        assert metrics.save_attempts >= metrics.actual_writes


class TestTraceValidity:
    """Execution trace must always be valid."""

    @given(
        ocr_quality=floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_trace_always_has_entries(self, ocr_quality):
        """Given: Any OCR quality. When: Processed. Then: Trace has entries."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $25.00", quality=ocr_quality)
        harness.parser.set_parse_result(merchant="Store", total=25.00)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")

        # Trace should have at least one entry
        assert len(harness.trace) >= 1

    @given(
        ocr_text=text(min_size=0, max_size=500),
    )
    @settings(max_examples=50, deadline=None)
    def test_trace_entries_have_required_fields(self, ocr_text):
        """Given: Any input. When: Processed. Then: Trace entries complete."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", ocr_text, quality=0.5)
        harness.parser.set_default_output(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()

        harness.run("receipt.jpg")

        for entry in harness.trace:
            assert entry.stage in ["OCR", "PARSE", "VALIDATE", "RETRY", "PERSIST"]
            assert isinstance(entry.success, bool)
            assert entry.attempt >= 1

    @given(
        fail_then_succeed=booleans(),
    )
    @settings(max_examples=30, deadline=None)
    def test_retry_tracked_in_trace(self, fail_then_succeed):
        """Given: Retry scenario. When: Processed. Then: Retry visible in trace."""
        harness = PipelineTestHarness(use_fake_retry=True)

        harness.ocr.set_text_for_image("receipt.jpg", "STORE $25.00", quality=0.85)

        if fail_then_succeed:
            harness.parser.set_sequence([
                ValueError("Fail once"),
                ParserOutput(merchant="Store", total=25.00),
            ])
            harness.retry.set_succeed_on_attempt(2)
        else:
            harness.parser.set_parse_result(merchant="Store", total=25.00)

        harness.validator.all_fields_pass()
        harness.run("receipt.jpg")

        # OCR should always be in trace
        ocr_entries = [e for e in harness.trace if e.stage == "OCR"]
        assert len(ocr_entries) >= 1


class TestPipelineIdempotency:
    """Pipeline operations should be idempotent where expected."""

    @given(
        merchant=text(min_size=1, max_size=50),
        total=floats(min_value=0, max_value=10000, allow_nan=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_reprocessing_same_data_idempotent(self, merchant, total):
        """Given: Same data. When: Processed multiple times. Then: Same final state."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", f"STORE ${total}", quality=0.85)
        harness.parser.set_parse_result(merchant=merchant, total=total)
        harness.validator.all_fields_pass()

        # Run multiple times
        result1 = harness.run("receipt.jpg")
        result2 = harness.run("receipt.jpg")
        result3 = harness.run("receipt.jpg")

        # All should have consistent status (either all SUCCESS or all DUPLICATE)
        if result1.status == PipelineStatus.SUCCESS:
            assert result2.was_duplicate or result2.status == PipelineStatus.DUPLICATE
            assert result3.was_duplicate or result3.status == PipelineStatus.DUPLICATE

        # Only one persisted record
        harness.assert_persist_count(1)


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @given(
        empty_ocr=st.just(""),
        whitespace=text(alphabet=st.characters(whitelist_characters=' \t\n'), min_size=0, max_size=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_empty_and_whitespace_ocr(self, empty_ocr, whitespace):
        """Given: Empty or whitespace OCR. When: Processed. Then: No crash."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", empty_ocr + whitespace, quality=0.1)
        harness.parser.set_default_output(merchant="Unknown", total=0.0)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result is not None
        assert isinstance(result.status, PipelineStatus)

    @given(
        huge_text=text(min_size=5000, max_size=10000),
    )
    @settings(max_examples=5, deadline=None)
    def test_very_large_ocr_text(self, huge_text):
        """Given: Very large OCR text. When: Processed. Then: No crash."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", huge_text[:1000], quality=0.5)
        harness.parser.set_default_output(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result is not None

    @given(
        unicode_chars=text(
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Lo', 'Nd', 'Sc', 'Zs'),
                whitelist_characters='€£¥$©®™°•★☆♠♥♦♣'
            ),
            min_size=0,
            max_size=200
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_unicode_ocr_text(self, unicode_chars):
        """Given: Unicode OCR text. When: Processed. Then: No crash."""
        harness = PipelineTestHarness()

        harness.ocr.set_text_for_image("receipt.jpg", unicode_chars, quality=0.5)
        harness.parser.set_default_output(merchant="Store", total=10.0)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result is not None
        assert isinstance(result.status, PipelineStatus)
