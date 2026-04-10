"""API response utilities"""

from typing import Any, Optional
from domain.models.receipt import Receipt as ReceiptModel


class APIResponse:
    """Standardized API response structure"""

    def __init__(self, status: str, data: Any = None, error: str = ""):
        self.status = status  # "success" or "failed"
        self.data = data
        self.error = error

    @classmethod
    def success(cls, data: Any) -> 'APIResponse':
        """Create a successful response"""
        return cls(status="success", data=data)

    @classmethod
    def failure(cls, error: str, data: Any = None) -> 'APIResponse':
        """Create a failed response"""
        return cls(status="failed", error=error, data=data)

    def validate_receipt_data(self) -> bool:
        """Validate that data is a proper ReceiptModel"""
        if not self.success or not self.data:
            return False

        return isinstance(self.data, ReceiptModel)
