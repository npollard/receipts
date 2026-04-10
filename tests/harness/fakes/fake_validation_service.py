"""Fake validation service for deterministic receipt validation testing."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import time

from .fake_component import FakeComponent, ConfigurationError


class ValidationField(Enum):
    """Fields that can be validated on a receipt."""
    MERCHANT = "merchant"
    DATE = "date"
    TOTAL = "total"
    ITEMS = "items"
    CURRENCY = "currency"


@dataclass
class FieldValidation:
    """Validation result for a single field."""
    field: str
    is_valid: bool
    error_message: Optional[str] = None
    normalized_value: Any = None


@dataclass
class ValidationResult:
    """Complete validation result for a receipt."""
    is_valid: bool
    field_results: Dict[str, FieldValidation]
    errors: List[str]
    warnings: List[str]
    preserve_partial: bool = True


class FakeValidationService(FakeComponent):
    """Fake validation service with configurable pass/fail behavior.
    
    Simulates:
    - Full validation pass
    - Field-level failures
    - Partial preservation of valid fields
    - Normalization (e.g., date formats)
    
    Example:
        >>> validator = FakeValidationService()
        >>> validator.field_passes(ValidationField.MERCHANT)
        >>> validator.field_fails(ValidationField.DATE, "Invalid format")
        >>> result = validator.validate_receipt({...})
        >>> result.is_valid  # False
        >>> result.field_results["date"].is_valid  # False
    """
    
    def __init__(self):
        super().__init__()
        self._field_configs: Dict[str, FieldValidation] = {}
        self._should_fail_all: bool = False
        self._global_error: Optional[str] = None
        self._preserve_partial: bool = True
        self._default_validator: Optional[Callable] = None
    
    def field_passes(
        self,
        field: ValidationField,
        normalized_value: Any = None
    ) -> "FakeValidationService":
        """Configure a field to pass validation.
        
        Args:
            field: Field to configure
            normalized_value: Optional normalized value to return
            
        Returns:
            Self for chaining
        """
        self._field_configs[field.value] = FieldValidation(
            field=field.value,
            is_valid=True,
            normalized_value=normalized_value
        )
        return self
    
    def field_fails(
        self,
        field: ValidationField,
        error_message: str = "Validation failed"
    ) -> "FakeValidationService":
        """Configure a field to fail validation.
        
        Args:
            field: Field to configure
            error_message: Error message for failure
            
        Returns:
            Self for chaining
        """
        self._field_configs[field.value] = FieldValidation(
            field=field.value,
            is_valid=False,
            error_message=error_message
        )
        return self
    
    def all_fields_pass(self) -> "FakeValidationService":
        """Configure all fields to pass (default success case)."""
        for field in ValidationField:
            self.field_passes(field)
        return self
    
    def all_fields_fail(self, error_message: str = "Validation failed") -> "FakeValidationService":
        """Configure all fields to fail."""
        self._should_fail_all = True
        self._global_error = error_message
        return self
    
    def set_preserve_partial(self, should_preserve: bool) -> "FakeValidationService":
        """Configure whether to preserve partial results."""
        self._preserve_partial = should_preserve
        return self
    
    def set_custom_validator(
        self,
        validator: Callable[[Dict[str, Any]], ValidationResult]
    ) -> "FakeValidationService":
        """Set a custom validation function."""
        self._default_validator = validator
        return self
    
    def reset(self) -> None:
        """Reset all field configurations."""
        super().reset()
        self._field_configs.clear()
        self._should_fail_all = False
        self._global_error = None
        self._preserve_partial = True
    
    def validate_receipt(self, receipt_data: Dict[str, Any]) -> ValidationResult:
        """Validate receipt data according to configuration.
        
        Args:
            receipt_data: Dictionary with receipt fields
            
        Returns:
            ValidationResult with pass/fail status per field
        """
        start = time.time()
        
        # Use custom validator if set
        if self._default_validator:
            result = self._default_validator(receipt_data)
            duration_ms = (time.time() - start) * 1000
            self._record_call(
                "validate_receipt",
                (receipt_data,),
                {},
                result=result,
                duration_ms=duration_ms
            )
            return result
        
        # Build validation result
        field_results: Dict[str, FieldValidation] = {}
        errors: List[str] = []
        warnings: List[str] = []
        
        if self._should_fail_all:
            # All fields fail
            for field in ValidationField:
                field_results[field.value] = FieldValidation(
                    field=field.value,
                    is_valid=False,
                    error_message=self._global_error or f"{field.value} validation failed"
                )
                errors.append(f"{field.value}: {self._global_error}")
            is_valid = False
        else:
            # Check each field against configuration
            is_valid = True
            for field in ValidationField:
                config = self._field_configs.get(field.value)
                if config:
                    field_results[field.value] = config
                    if not config.is_valid:
                        is_valid = False
                        errors.append(f"{field.value}: {config.error_message}")
                else:
                    # No config = pass by default
                    field_results[field.value] = FieldValidation(
                        field=field.value,
                        is_valid=True
                    )
        
        result = ValidationResult(
            is_valid=is_valid,
            field_results=field_results,
            errors=errors,
            warnings=warnings,
            preserve_partial=self._preserve_partial
        )
        
        duration_ms = (time.time() - start) * 1000
        self._record_call(
            "validate_receipt",
            (receipt_data,),
            {},
            result=result,
            duration_ms=duration_ms
        )
        
        return result
    
    def validate_field(
        self,
        field_name: str,
        value: Any,
        rules: Optional[List[str]] = None
    ) -> FieldValidation:
        """Validate a single field.
        
        Args:
            field_name: Name of the field
            value: Value to validate
            rules: Optional list of validation rule names
            
        Returns:
            FieldValidation result
        """
        start = time.time()
        
        config = self._field_configs.get(field_name)
        if config:
            result = config
        else:
            # Default: pass
            result = FieldValidation(
                field=field_name,
                is_valid=True,
                normalized_value=value
            )
        
        duration_ms = (time.time() - start) * 1000
        self._record_call(
            "validate_field",
            (field_name, value),
            {"rules": rules},
            result=result,
            duration_ms=duration_ms
        )
        
        return result
    
    # Assertion helpers
    
    def get_validated_fields(self) -> List[str]:
        """Get list of field names that were validated."""
        fields = set()
        for call in self.get_calls("validate_receipt"):
            if call.result and isinstance(call.result, ValidationResult):
                fields.update(call.result.field_results.keys())
        return list(fields)
    
    def get_failed_fields(self) -> List[str]:
        """Get list of field names that failed validation."""
        failed = []
        for call in self.get_calls("validate_receipt"):
            if call.result and isinstance(call.result, ValidationResult):
                for field, result in call.result.field_results.items():
                    if not result.is_valid:
                        failed.append(field)
        return failed
    
    def was_field_validated(self, field: ValidationField) -> bool:
        """Check if a specific field was validated."""
        return field.value in self.get_validated_fields()
    
    def did_field_fail(self, field: ValidationField) -> bool:
        """Check if a specific field failed validation."""
        return field.value in self.get_failed_fields()
