"""Fake implementations for deterministic testing.

Fakes replace external dependencies (OCR, LLM, DB) with configurable,
in-memory implementations that track calls for assertions.
"""

from .fake_component import FakeComponent, CallRecord
from .fake_ocr_service import FakeOCRService, OCROutput
from .fake_receipt_parser import FakeReceiptParser, ParserOutput
from .fake_validation_service import FakeValidationService, ValidationResult
from .fake_repository import FakeRepository
from .fake_retry_service import FakeRetryService

__all__ = [
    "FakeComponent",
    "CallRecord",
    "FakeOCRService",
    "OCROutput",
    "FakeReceiptParser",
    "ParserOutput",
    "FakeValidationService",
    "ValidationResult",
    "FakeRepository",
    "FakeRetryService",
]
