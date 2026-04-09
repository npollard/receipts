#!/usr/bin/env python3
"""Fast integration test with dependency-injected fakes (no EasyOCR)"""

from tests.fakes.fake_vision_processor import FakeVisionProcessor
from tests.fakes.fake_receipt_parser import FakeReceiptParser
from services.batch_service import BatchProcessingService


def test_integrated_pipeline_with_fakes():
    """Test integrated pipeline with DI fakes - runs in milliseconds without EasyOCR"""

    # Create fake dependencies (no imports of image_processing or EasyOCR)
    image_processor = FakeVisionProcessor(
        text="MILK 4.50\nBREAD 3.00\nTOTAL 7.50"
    )

    receipt_parser = FakeReceiptParser({
        "date": "2026-03-30",
        "total": 7.50,
        "items": [
            {"description": "Milk", "price": 4.50},
            {"description": "Bread", "price": 3.00}
        ],
        "_token_usage": {
            "input_tokens": 25,
            "output_tokens": 15
        }
    })

    batch_service = BatchProcessingService()

    # Test with dummy paths (fakes don't need real files)
    image_paths = ["/dummy/path/receipt1.jpg", "/dummy/path/receipt2.jpg"]

    # Run the pipeline - completes in milliseconds
    successful, failed, token_usage, _ = batch_service.process_batch(
        image_paths, image_processor, receipt_parser
    )

    # Validate results
    assert successful == 2, f"Expected 2 successful, got {successful}"
    assert failed == 0, f"Expected 0 failed, got {failed}"
    assert token_usage.input_tokens == 50, f"Expected 50 input tokens, got {token_usage.input_tokens}"
    assert token_usage.output_tokens == 30, f"Expected 30 output tokens, got {token_usage.output_tokens}"


def test_pipeline_with_mixed_success_and_failure():
    """Test pipeline handles mixed success/failure with fakes"""

    image_processor = FakeVisionProcessor(text="RECEIPT TEXT")

    # Create fake that fails on second call
    receipt_parser = FakeReceiptParser({
        "date": "2026-03-30",
        "total": 10.00,
        "items": [{"description": "Item", "price": 10.00}],
        "_token_usage": {
            "input_tokens": 20,
            "output_tokens": 10
        }
    })

    # Override to track calls and fail on second
    original_parse = receipt_parser.parse_with_validation_driven_retry
    call_count = [0]

    def conditional_parse(ocr_text, image_path=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return original_parse(ocr_text, image_path)
        else:
            from domain.parsing.receipt_parser import ParsingResult
            result = ParsingResult()
            result.parsed = None
            result.valid = False
            result.error = "Validation failed"
            return result

    receipt_parser.parse_with_validation_driven_retry = conditional_parse

    batch_service = BatchProcessingService()
    image_paths = ["/dummy/path/receipt1.jpg", "/dummy/path/receipt2.jpg"]

    successful, failed, token_usage, _ = batch_service.process_batch(
        image_paths, image_processor, receipt_parser
    )

    assert successful == 1
    assert failed == 1
    assert token_usage.input_tokens == 20  # Only successful parse counts
