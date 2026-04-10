"""Builder for constructing and configuring test harness."""

from typing import Optional, List, Dict, Any
from tests.harness.fakes import (
    FakeOCRService,
    FakeReceiptParser,
    FakeValidationService,
    FakeRepository,
    FakeRetryService,
)


class HarnessBuilder:
    """Builder for constructing a configured test harness.
    
    Provides a fluent API for setting up all fake components
    with common test scenarios.
    
    Example:
        >>> harness = (HarnessBuilder()
        ...     .with_ocr_text("receipt.jpg", "STORE $10")
        ...     .with_parsed_receipt(merchant="Store", total=10.0)
        ...     .with_validation_passing()
        ...     .build())
    """
    
    def __init__(self):
        self._ocr = FakeOCRService()
        self._parser = FakeReceiptParser()
        self._validator = FakeValidationService()
        self._repository = FakeRepository()
        self._retry = FakeRetryService()
    
    # OCR configuration
    
    def with_ocr_text(
        self,
        image_path: str,
        text: str,
        quality: float = 1.0
    ) -> "HarnessBuilder":
        """Configure OCR to return text for an image."""
        self._ocr.set_text_for_image(image_path, text, quality)
        return self
    
    def with_ocr_failing(
        self,
        exception: Exception,
        max_attempts: int = 1
    ) -> "HarnessBuilder":
        """Configure OCR to fail with given exception."""
        self._ocr.set_should_fail(exception, max_attempts)
        return self
    
    def with_ocr_fallback(
        self,
        image_path: str,
        text: str,
        quality: float = 1.0
    ) -> "HarnessBuilder":
        """Configure OCR fallback (Vision API) output."""
        self._ocr.set_fallback_output(image_path, text, quality)
        return self
    
    # Parser configuration
    
    def with_parsed_receipt(
        self,
        merchant: str = "",
        total: Optional[float] = None,
        date: str = "",
        items: Optional[List[Dict[str, Any]]] = None
    ) -> "HarnessBuilder":
        """Configure parser to return successful parse."""
        self._parser.set_parse_result(
            merchant=merchant,
            total=total,
            date=date,
            items=items
        )
        return self
    
    def with_parser_failing(self, exception: Exception) -> "HarnessBuilder":
        """Configure parser to fail."""
        self._parser.set_parse_fails_with(exception)
        return self
    
    def with_parser_retries_then_succeeds(
        self,
        fail_times: int,
        merchant: str = "",
        total: float = 0.0
    ) -> "HarnessBuilder":
        """Configure parser to fail N times then succeed."""
        # First N outputs are failures
        for _ in range(fail_times):
            self._parser.set_parse_fails_with(ValueError("Parse failed"))
        # Final output is success
        self._parser.set_parse_result(merchant=merchant, total=total)
        return self
    
    # Validation configuration
    
    def with_validation_passing(self) -> "HarnessBuilder":
        """Configure all validations to pass."""
        self._validator.all_fields_pass()
        return self
    
    def with_validation_failing(
        self,
        field: str,
        error: str = "Validation failed"
    ) -> "HarnessBuilder":
        """Configure specific field to fail validation."""
        from tests.harness.fakes.fake_validation_service import ValidationField
        field_enum = ValidationField(field)
        self._validator.field_fails(field_enum, error)
        return self
    
    # Repository configuration
    
    def with_existing_receipt(self, receipt_dto: Any) -> "HarnessBuilder":
        """Pre-populate repository with a receipt."""
        self._repository.seed_receipt(receipt_dto)
        return self
    
    def with_existing_user(self, user_dto: Any) -> "HarnessBuilder":
        """Pre-populate repository with a user."""
        self._repository.seed_user(user_dto)
        return self
    
    def with_repository_failing_on(
        self,
        operation: str,
        exception: Exception
    ) -> "HarnessBuilder":
        """Configure repository to fail on specific operation."""
        self._repository.set_should_fail_on(operation, exception)
        return self
    
    # Retry configuration
    
    def with_retry_succeeding_first_time(self) -> "HarnessBuilder":
        """Configure retry to succeed on first attempt."""
        self._retry.set_succeed_on_attempt(1)
        return self
    
    def with_retry_succeeding_on_attempt(self, n: int) -> "HarnessBuilder":
        """Configure retry to succeed on Nth attempt."""
        self._retry.set_succeed_on_attempt(n)
        return self
    
    def with_retry_strategy_for_error(
        self,
        error_type: type,
        strategy: str
    ) -> "HarnessBuilder":
        """Configure retry strategy for specific error type."""
        self._retry.set_strategy_for_error(error_type, strategy)
        return self
    
    # Build
    
    def build(self) -> Dict[str, Any]:
        """Build and return configured fakes as dictionary."""
        return {
            "ocr": self._ocr,
            "parser": self._parser,
            "validator": self._validator,
            "repository": self._repository,
            "retry": self._retry,
        }
    
    def build_ocr_only(self) -> FakeOCRService:
        """Build and return only the OCR fake."""
        return self._ocr
    
    def build_parser_only(self) -> FakeReceiptParser:
        """Build and return only the parser fake."""
        return self._parser
