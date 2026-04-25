"""API response utilities"""

from typing import Any
from domain.models.receipt import Receipt as ReceiptModel


class _SuccessDescriptor:
    """Support APIResponse.success(data) and response.success checks."""

    def __get__(self, instance, owner):
        if instance is None:
            def factory(data: Any) -> "APIResponse":
                return owner(status="success", data=data)
            return factory
        return instance.status == "success"


class APIResponse:
    """Standardized API response structure"""

    success = _SuccessDescriptor()

    def __init__(self, status: str, data: Any = None, error: str = ""):
        self.status = status  # "success" or "failed"
        self.data = data
        self.error = error

    @classmethod
    def failure(cls, error: str, data: Any = None) -> 'APIResponse':
        """Create a failed response"""
        return cls(status="failed", error=error, data=data)

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    @property
    def is_failure(self) -> bool:
        return not self.is_success

    def validate_receipt_data(self) -> bool:
        """Validate that data is a proper ReceiptModel"""
        if not self.is_success or not self.data:
            return False

        return isinstance(self.data, ReceiptModel)
