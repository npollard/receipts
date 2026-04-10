"""Idempotency behavior tests using PipelineTestHarness.

Verifies that:
- Retries don't create duplicate records
- Identical inputs result in single persistence
- Distinct inputs result in separate records
- Hash collisions are handled correctly
"""

import pytest
from tests.harness.pipeline_harness import PipelineTestHarness, PipelineStatus
from tests.harness.builders import ReceiptBuilder


class TestRetryDoesNotDuplicate:
    """Verify retry attempts don't create duplicate records."""

    def test_retry_does_not_duplicate_persistence(self):
        """Given: Parser fails twice then succeeds.
        When: Pipeline runs with retry.
        Then: Only one receipt persisted despite 3 attempts.
        """
        harness = PipelineTestHarness(use_fake_retry=True)

        # Configure OCR
        harness.ocr.set_text_for_image("receipt.jpg", "STORE $25.00", quality=0.85)

        # Parser fails twice, succeeds on third attempt
        harness.parser.set_sequence([
            ValueError("Missing date field"),
            ValueError("Invalid JSON"),
            ParserOutput(merchant="Store", total=25.00, date="2024-01-15"),
        ])

        # Retry configured for 3 attempts
        harness.retry.set_succeed_on_attempt(3)
        harness.retry.set_max_retries(3)
        harness.validator.all_fields_pass()

        # Execute
        result = harness.run("receipt.jpg")

        # Verify success
        assert result.status == PipelineStatus.SUCCESS

        # Assert retry behavior
        assert harness.retry.get_attempt_count() == 3

        # CRITICAL: Only 1 actual write despite 3 parse attempts
        harness.assert_persist_count(1)
        harness.assert_no_duplicate_persistence()

        # Verify metrics
        metrics = harness.get_repository_metrics()
        assert metrics.save_attempts == 1  # Called save once
        assert metrics.actual_writes == 1  # Only 1 write occurred
        assert metrics.duplicate_detections == 0  # No duplicates

    def test_ocr_retry_then_success_single_persist(self):
        """Given: OCR fails twice, succeeds on third.
        When: Processed with OCR retry.
        Then: Single receipt persisted.
        """
        harness = PipelineTestHarness()

        # OCR: fail, fail, success
        harness.ocr.set_sequence([
            OCRError("Network timeout"),
            OCRError("Service unavailable"),
            OCROutput(text="STORE $50.00", quality_score=0.88),
        ])

        harness.parser.set_parse_result(merchant="Store", total=50.00)
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status == PipelineStatus.SUCCESS

        # OCR made 3 attempts
        ocr_history = harness.ocr.get_attempt_history()
        assert len(ocr_history) == 3

        # But only 1 receipt persisted
        harness.assert_persist_count(1)
        harness.assert_unique_hashes(1)


class TestSameInputSinglePersistence:
    """Verify identical inputs result in single persisted record."""

    def test_same_input_multiple_runs_only_persists_once(self):
        """Given: Same receipt processed 3 times.
        When: Each run completes.
        Then: Only 1 record in repository.
        """
        harness = PipelineTestHarness()

        # Configure for success
        harness.ocr.set_text_for_image("receipt.jpg", "STORE $30.00", quality=0.90)
        harness.parser.set_parse_result(
            merchant="Store",
            total=30.00,
            date="2024-03-15"
        )
        harness.validator.all_fields_pass()

        # Run 3 times with identical input
        result1 = harness.run("receipt.jpg")
        result2 = harness.run("receipt.jpg")
        result3 = harness.run("receipt.jpg")

        # First succeeds, subsequent detected as duplicates
        assert result1.status == PipelineStatus.SUCCESS
        # Pipeline detects duplicate by image hash before save
        assert result2.was_duplicate or result2.status == PipelineStatus.DUPLICATE
        assert result3.was_duplicate or result3.status == PipelineStatus.DUPLICATE

        # But repository has only 1 record
        harness.assert_persist_count(1)
        harness.assert_unique_hashes(1)

        # Verify only 1 actual write occurred
        metrics = harness.get_repository_metrics()
        assert metrics.actual_writes == 1  # Only 1 write

        # Verify persisted records
        records = harness.get_persisted_records()
        assert len(records) == 1
        assert records[0].total_amount == 30.00

    def test_same_content_different_image_paths_single_record(self):
        """Given: Same receipt content, different image paths.
        When: Processed.
        Then: Single record (deduplicated by content hash).
        """
        harness = PipelineTestHarness(use_fake_retry=True)

        # Same content, different image paths (different image hashes)
        harness.ocr.set_text_for_image("receipt1.jpg", "STORE $40.00", quality=0.85)
        harness.ocr.set_text_for_image("receipt2.jpg", "STORE $40.00", quality=0.85)

        # Same parse result for both -> same content hash
        # Configure parser with sequence for both receipts
        harness.parser.set_sequence([
            ParserOutput(merchant="Store", total=40.00),
            ParserOutput(merchant="Store", total=40.00),
        ])
        harness.validator.all_fields_pass()

        # Process both
        result1 = harness.run("receipt1.jpg")
        result2 = harness.run("receipt2.jpg")

        # Both succeed (different images, but content deduplicated at repository)
        assert result1.status == PipelineStatus.SUCCESS
        assert result2.status == PipelineStatus.SUCCESS

        # Single record persisted (content hash deduplication in repository)
        harness.assert_persist_count(1)
        harness.assert_unique_hashes(1)

        # Verify repository detected duplicate by content hash
        metrics = harness.get_repository_metrics()
        assert metrics.save_attempts == 2  # 2 save attempts
        assert metrics.actual_writes == 1  # But only 1 write (2nd was duplicate)


class TestDifferentInputsDistinctRecords:
    """Verify distinct inputs result in separate records."""

    def test_slightly_different_inputs_produce_distinct_records(self):
        """Given: Three different receipts (different totals).
        When: Each processed.
        Then: 3 distinct records in repository.
        """
        harness = PipelineTestHarness()
        harness.validator.all_fields_pass()

        # Receipt 1: $25.00
        harness.ocr.set_text_for_image("receipt1.jpg", "STORE A $25.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=25.00)
        result1 = harness.run("receipt1.jpg")

        # Receipt 2: $50.00 (different total = different hash)
        harness.ocr.set_text_for_image("receipt2.jpg", "STORE B $50.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=50.00)
        result2 = harness.run("receipt2.jpg")

        # Receipt 3: $75.00 (different total = different hash)
        harness.ocr.set_text_for_image("receipt3.jpg", "STORE C $75.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=75.00)
        result3 = harness.run("receipt3.jpg")

        # All succeed
        assert result1.status == PipelineStatus.SUCCESS
        assert result2.status == PipelineStatus.SUCCESS
        assert result3.status == PipelineStatus.SUCCESS

        # 3 distinct records (different totals = different content hashes)
        harness.assert_persist_count(3)
        harness.assert_unique_hashes(3)

        # Verify persisted records
        records = harness.get_persisted_records()
        assert len(records) == 3
        totals = {r.total_amount for r in records}
        assert totals == {25.00, 50.00, 75.00}

    def test_same_merchant_different_totals_distinct_records(self):
        """Given: Same merchant, different totals.
        When: Processed.
        Then: Separate records for each total (different content hashes).
        """
        harness = PipelineTestHarness()
        harness.validator.all_fields_pass()

        # Three receipts from same store, different amounts
        for image, total in [
            ("r1.jpg", 10.00),
            ("r2.jpg", 20.00),
            ("r3.jpg", 30.00),
        ]:
            harness.ocr.set_text_for_image(image, f"GROCERY ${total}", quality=0.85)
            harness.parser.set_parse_result(merchant="Grocery Store", total=total)

            result = harness.run(image)
            assert result.status == PipelineStatus.SUCCESS

        # 3 distinct records (different totals = different hashes)
        harness.assert_persist_count(3)
        harness.assert_unique_hashes(3)


class TestHashCollisionHandling:
    """Verify hash collision scenarios are handled correctly."""

    def test_hash_collision_protection_behavior(self):
        """Given: Two identical receipts processed.
        When: Second receipt saved.
        Then: Duplicate detection prevents second write.
        """
        harness = PipelineTestHarness()

        # First receipt: normal save
        harness.ocr.set_text_for_image("receipt1.jpg", "STORE $100.00", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=100.00)
        harness.validator.all_fields_pass()

        result1 = harness.run("receipt1.jpg")
        assert result1.status == PipelineStatus.SUCCESS

        # Second receipt with identical content
        harness.ocr.set_text_for_image("receipt2.jpg", "DIFFERENT TEXT BUT SAME PARSE", quality=0.85)
        harness.parser.set_parse_result(merchant="Store", total=100.00)  # Same content = same hash

        # Duplicate detection prevents second write
        result2 = harness.run("receipt2.jpg")

        # Either SUCCESS (as duplicate) or DUPLICATE status
        assert result2.status in [PipelineStatus.SUCCESS, PipelineStatus.DUPLICATE]

        # Only 1 record persisted
        harness.assert_persist_count(1)
        harness.assert_unique_hashes(1)

        # Metrics show duplicate detection
        metrics = harness.get_repository_metrics()
        assert metrics.actual_writes == 1
        assert metrics.duplicate_detections == 1

    def test_explicit_different_hashes_force_distinct_records(self):
        """Given: Two receipts with different content.
        When: Processed.
        Then: Always 2 distinct records (different content hashes).
        """
        harness = PipelineTestHarness()
        harness.validator.all_fields_pass()

        # First receipt: Store A, $10.00
        harness.ocr.set_text_for_image("r1.jpg", "STORE A $10", quality=0.85)
        harness.parser.set_parse_result(merchant="Store A", total=10.00)
        result1 = harness.run("r1.jpg")

        # Second receipt: Store B, $20.00 (different = different hash)
        harness.ocr.set_text_for_image("r2.jpg", "STORE B $20", quality=0.85)
        harness.parser.set_parse_result(merchant="Store B", total=20.00)
        result2 = harness.run("r2.jpg")

        assert result1.status == PipelineStatus.SUCCESS
        assert result2.status == PipelineStatus.SUCCESS

        # 2 distinct records (different content = different hashes)
        harness.assert_persist_count(2)
        harness.assert_unique_hashes(2)


# Import needed for type hints
from tests.harness.fakes import OCROutput
from tests.harness.fakes.fake_receipt_parser import ParserOutput
from core.exceptions import OCRError, DataIntegrityError
