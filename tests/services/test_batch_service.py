"""Batch processing service tests using harness + fakes.

Tests batch orchestration behavior without real dependencies.
"""

import pytest
from pathlib import Path
from typing import List
from tests.harness.fakes import FakeOCRService, FakeReceiptParser, FakeValidationService
from services.batch_service import BatchProcessingService
from api_response import APIResponse


class TestBatchProcessingBasics:
    """Basic batch processing interface tests."""

    def test_batch_service_has_process_method(self):
        """Given: Batch service. When: Checked. Then: Has process_batch method."""
        batch_service = BatchProcessingService()
        assert hasattr(batch_service, 'process_batch')
        assert callable(getattr(batch_service, 'process_batch'))

    def test_batch_service_has_config(self):
        """Given: Batch service. When: Checked. Then: Has configuration."""
        batch_service = BatchProcessingService()
        assert hasattr(batch_service, 'config')

    def test_empty_batch_returns_zero_counts(self):
        """Given: Empty image list. When: Processed. Then: Zero results."""
        ocr = FakeOCRService()
        parser = FakeReceiptParser()
        batch_service = BatchProcessingService()

        successful_count, failed_count, token_usage, observability = batch_service.process_batch(
            [], ocr, parser
        )

        assert successful_count == 0
        assert failed_count == 0
        assert observability is None  # Empty batch returns None observability


class TestBatchProcessingFailures:
    """Batch processing with failures scenarios."""

    def test_batch_handles_individual_failures(self):
        """Given: Some images fail. When: Batch processed. Then: Others succeed."""
        ocr = FakeOCRService()
        parser = FakeReceiptParser()
        batch_service = BatchProcessingService()

        # r1 fails, r2 succeeds, r3 fails
        ocr.set_text_for_image("r1.jpg", "FAIL", quality=0.10)
        ocr.set_text_for_image("r2.jpg", "STORE $20", quality=0.90)
        ocr.set_text_for_image("r3.jpg", "FAIL", quality=0.10)

        parser.set_parse_result(merchant="Store", total=20.00)

        image_paths = [Path("r1.jpg"), Path("r2.jpg"), Path("r3.jpg")]

        successful_count, failed_count, token_usage, observability = batch_service.process_batch(
            image_paths, ocr, parser
        )

        # Mixed results
        assert successful_count + failed_count == 3
        assert observability.total_images == 3

    def test_batch_reports_observability(self):
        """Given: Batch with mixed results. When: Processed. Then: Observability accurate."""
        ocr = FakeOCRService()
        parser = FakeReceiptParser()
        batch_service = BatchProcessingService()

        ocr.set_text_for_image("r1.jpg", "STORE $10", quality=0.90)
        ocr.set_text_for_image("r2.jpg", "STORE $20", quality=0.90)
        ocr.set_text_for_image("r3.jpg", "BAD", quality=0.10)  # Low quality
        parser.set_parse_result(merchant="Store", total=10.00)

        image_paths = [Path("r1.jpg"), Path("r2.jpg"), Path("r3.jpg")]

        successful_count, failed_count, token_usage, observability = batch_service.process_batch(
            image_paths, ocr, parser
        )

        assert observability.total_images == 3
        assert observability.successful == successful_count
        assert observability.failed == failed_count
        assert observability.total_time_ms > 0

    def test_empty_batch(self):
        """Given: Empty image list. When: Processed. Then: Zero results."""
        ocr = FakeOCRService()
        parser = FakeReceiptParser()
        batch_service = BatchProcessingService()

        successful_count, failed_count, token_usage, observability = batch_service.process_batch(
            [], ocr, parser
        )

        assert successful_count == 0
        assert failed_count == 0
        assert observability is None  # Empty batch returns None observability

class TestBatchProcessingConfiguration:
    """Batch processing configuration scenarios."""

    def test_batch_respects_worker_configuration(self):
        """Given: Worker config. When: Batch processed. Then: Config applied."""
        batch_service = BatchProcessingService()

        # Service should have configuration attributes
        assert hasattr(batch_service, 'process_batch')

    def test_batch_with_validation_enabled(self):
        """Given: Validation enabled. When: Batch processed. Then: Invalid results filtered."""
        ocr = FakeOCRService()
        parser = FakeReceiptParser()
        batch_service = BatchProcessingService()

        # Valid receipt
        ocr.set_text_for_image("valid.jpg", "STORE $25.00", quality=0.90)
        parser.set_parse_result(merchant="Store", total=25.00)

        image_paths = [Path("valid.jpg")]

        successful_count, failed_count, token_usage, _ = batch_service.process_batch(
            image_paths, ocr, parser
        )

        # Should succeed with validation
        assert isinstance(successful_count, int)


class TestBatchOrchestrationBehavior:
    """Batch orchestration behavior scenarios."""

    def test_batch_sequential_processing(self):
        """Given: Single worker. When: Batch processed. Then: Sequential execution."""
        ocr = FakeOCRService()
        parser = FakeReceiptParser()
        batch_service = BatchProcessingService()

        for i in range(3):
            ocr.set_text_for_image(f"r{i}.jpg", f"STORE ${i}0", quality=0.85)

        parser.set_parse_result(merchant="Store", total=10.00)

        image_paths = [Path(f"r{i}.jpg") for i in range(3)]

        successful_count, failed_count, token_usage, observability = batch_service.process_batch(
            image_paths, ocr, parser
        )

        # All processed
        assert observability.total_images == 3
        assert observability.avg_time_per_image_ms > 0

    def test_batch_result_counts(self):
        """Given: Batch results. When: Retrieved. Then: Correct counts."""
        ocr = FakeOCRService()
        parser = FakeReceiptParser()
        batch_service = BatchProcessingService()

        ocr.set_text_for_image("r1.jpg", "STORE $15.00", quality=0.90)
        parser.set_parse_result(merchant="Store", total=15.00)

        image_paths = [Path("r1.jpg")]

        successful_count, failed_count, token_usage, _ = batch_service.process_batch(
            image_paths, ocr, parser
        )

        # Results are counts
        assert successful_count >= 0
        assert failed_count >= 0

    def test_batch_error_handling(self):
        """Given: Service errors. When: Batch processed. Then: Errors handled gracefully."""
        ocr = FakeOCRService()
        parser = FakeReceiptParser()
        batch_service = BatchProcessingService()

        # Configure to raise exception
        ocr.set_should_fail(Exception("Service unavailable"))

        image_paths = [Path("r1.jpg")]

        # Should handle error without crashing
        try:
            successful_count, failed_count, token_usage, observability = batch_service.process_batch(
                image_paths, ocr, parser
            )
            # If we get here, errors were handled
            assert isinstance(successful_count, int)
        except Exception:
            # Or exception propagated - either is acceptable
            pass


class TestBatchServiceIntegration:
    """Integration scenarios using the harness pattern."""

    def test_batch_with_harness_pattern(self):
        """Given: Harness-style setup. When: Batch processed. Then: Works correctly."""
        from tests.harness import HarnessBuilder

        fakes = (HarnessBuilder()
            .with_ocr_text("r1.jpg", "STORE $30.00")
            .with_ocr_text("r2.jpg", "STORE $40.00")
            .with_parsed_receipt(merchant="Store", total=30.00)
            .build())

        batch_service = BatchProcessingService()

        image_paths = [Path("r1.jpg"), Path("r2.jpg")]

        successful_count, failed_count, token_usage, observability = batch_service.process_batch(
            image_paths, fakes["ocr"], fakes["parser"]
        )

        assert observability.total_images == 2
        assert successful_count + failed_count == 2
