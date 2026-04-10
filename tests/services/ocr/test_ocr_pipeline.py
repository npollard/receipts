"""Consolidated OCR pipeline tests.

Covers: text extraction, image processing, field extraction, debug functionality.
Replaces: test_debug_ocr.py, test_field_extraction.py, test_image_processing.py
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from services.ocr_service import OCRService
from config.ocr_config import OCRConfig


class TestTextExtraction:
    """OCR text extraction behavior."""

    def test_extracts_text_from_receipt(self):
        """Given: Receipt text input. When: Processed. Then: Text returned."""
        service = OCRService(config=OCRConfig())

        # Test with text directly (simulating OCR output)
        receipt_text = "STORE\nItem $5.00\nTotal: $15.99"

        # Text should be processed without error
        result = service.extract_receipt_fields(receipt_text)

        assert result['total_amount'] == 15.99
        assert len(result['items']) > 0


class TestImageProcessing:
    """Image preprocessing behavior."""

    def test_image_input_processed_for_extraction(self):
        """Given: Image array. When: Preprocessed. Then: Ready for extraction."""
        service = OCRService(config=OCRConfig())

        # Create valid image array
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Should process without error and return array
        result = service._preprocess_image(dummy_image)
        assert isinstance(result['image_array'], np.ndarray)

    def test_empty_image_rejected(self):
        """Given: Empty image. When: Processed. Then: Error raised."""
        service = OCRService(config=OCRConfig())

        empty_image = np.array([])

        with pytest.raises(Exception):
            service._preprocess_image(empty_image)


class TestFieldExtraction:
    """Extract structured fields from OCR text."""

    def test_receipt_fields_extracted_from_text(self):
        """Given: OCR text. When: Processed. Then: Fields extracted."""
        service = OCRService(config=OCRConfig())

        text = "STORE\nItem $5.00\nTotal: $15.99"
        result = service.extract_receipt_fields(text)

        assert result['total_amount'] == 15.99
        assert len(result['items']) >= 1

    def test_empty_text_returns_safe_defaults(self):
        """Given: Empty text. When: Processed. Then: Safe defaults returned."""
        service = OCRService(config=OCRConfig())

        result = service.extract_receipt_fields("")

        assert result['total_amount'] is None
        assert result['items'] == []
        assert result['confidence'] == 0.0


class TestDebugFunctionality:
    """Debug mode features."""

    def test_debug_mode_configurable(self):
        """Given: Debug config. When: Service created. Then: Debug mode set correctly."""
        debug_service = OCRService(config=OCRConfig(debug_ocr=True))
        normal_service = OCRService(config=OCRConfig(debug_ocr=False))

        assert debug_service.debug_ocr is True
        assert normal_service.debug_ocr is False


class TestOCRConfiguration:
    """OCR service configuration."""

    def test_default_configuration_applied(self):
        """Given: No config provided. When: Service created. Then: Defaults used."""
        service = OCRService()

        assert service.quality_threshold == 0.25  # Default from config
        assert service.use_gpu is False

    def test_custom_configuration_applied(self):
        """Given: Custom config. When: Service created. Then: Custom values used."""
        config = OCRConfig(
            quality_threshold=0.5,
            use_gpu=False,
            languages=['en', 'fr']
        )
        service = OCRService(config=config)

        assert service.quality_threshold == 0.5
        assert service.lang == ['en', 'fr']

    def test_parameter_override_config(self):
        """Given: Config + parameters. When: Service created. Then: Parameters override."""
        config = OCRConfig(quality_threshold=0.3)
        service = OCRService(config=config, quality_threshold=0.7)

        assert service.quality_threshold == 0.7


class TestOCRErrorHandling:
    """Error handling in OCR operations."""

    def test_service_handles_invalid_inputs(self):
        """Given: Invalid inputs. When: Processed. Then: Handled gracefully."""
        service = OCRService(config=OCRConfig())

        # Empty text should give zero score
        score = service.score_ocr_quality("")
        assert score == 0.0

    def test_empty_text_returns_zero_quality(self):
        """Given: Empty text. When: Scored. Then: Zero quality."""
        service = OCRService(config=OCRConfig())

        score = service.score_ocr_quality("")

        assert score == 0.0
