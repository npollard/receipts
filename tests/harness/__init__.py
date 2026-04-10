"""Test harness for deterministic, composable testing.

This package provides fake implementations of all external dependencies
(OCR, LLM, database) with call tracking for assertions.

Quick Start:
    >>> from tests.harness.fakes import FakeOCRService, FakeRepository
    >>> from tests.harness.builders import ReceiptBuilder
    >>> 
    >>> ocr = FakeOCRService()
    >>> ocr.set_text_for_image("receipt.jpg", "STORE $10")
    >>> 
    >>> repo = FakeRepository()
    >>> repo.seed_receipt(ReceiptBuilder().with_total(10.0).build())

Modules:
    fakes: Fake implementations with call tracking
    builders: Fluent builders for test data

See EXAMPLES.md for detailed usage patterns.
"""

from .fakes import (
    FakeComponent,
    CallRecord,
    FakeOCRService,
    OCROutput,
    FakeReceiptParser,
    ParserOutput,
    FakeValidationService,
    ValidationResult,
    FakeRepository,
    FakeRetryService,
)

from .builders import (
    ReceiptBuilder,
    OCRResultBuilder,
    HarnessBuilder,
)

__all__ = [
    # Fakes
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
    # Builders
    "ReceiptBuilder",
    "OCRResultBuilder",
    "HarnessBuilder",
]
