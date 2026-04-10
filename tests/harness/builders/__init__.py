"""Fluent builders for constructing test data."""

from .receipt_builder import ReceiptBuilder
from .ocr_result_builder import OCRResultBuilder
from .harness_builder import HarnessBuilder
from .scenarios import (
    make_happy_path,
    make_easyocr_fail_then_vision_success,
    make_validation_failure_then_retry_success,
    make_duplicate_detected,
    make_partial_save_with_validation_errors,
    make_full_failure_ocr_error,
    make_full_failure_validation_critical,
    make_retry_exhausted_all_attempts_fail,
    make_complex_fallback_and_retry,
    ScenarioBuilder,
)

__all__ = [
    "ReceiptBuilder",
    "OCRResultBuilder",
    "HarnessBuilder",
    # Scenarios
    "make_happy_path",
    "make_easyocr_fail_then_vision_success",
    "make_validation_failure_then_retry_success",
    "make_duplicate_detected",
    "make_partial_save_with_validation_errors",
    "make_full_failure_ocr_error",
    "make_full_failure_validation_critical",
    "make_retry_exhausted_all_attempts_fail",
    "make_complex_fallback_and_retry",
    "ScenarioBuilder",
]
