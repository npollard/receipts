"""Domain validation module for receipt processing"""

from .validation_service import ValidationService
from .validation_utils import (
    validate_response_content,
    validate_with_pydantic,
    handle_validation_error,
    has_meaningful_receipt_data,
)
from .ocr_quality import (
    score_ocr_quality,
    should_fallback,
    get_fallback_reasoning,
)

__all__ = [
    "ValidationService",
    "validate_response_content",
    "validate_with_pydantic",
    "handle_validation_error",
    "has_meaningful_receipt_data",
    "score_ocr_quality",
    "should_fallback",
    "get_fallback_reasoning",
]
