"""API response utilities"""

from typing import Any, Optional
from models.receipt import Receipt as ReceiptModel


class APIResponse:
    """Standardized API response structure"""

    def __init__(self, status: str, data: Any = None, error: str = ""):
        self.status = status  # "success" or "failed"
        self.data = data
        self.error = error

    @classmethod
    def success(cls, data: Any) -> 'APIResponse':
        """Create a successful response"""
        # If data is already a ReceiptModel, use it directly
        if isinstance(data, ReceiptModel):
            return cls(status="success", data=data)

        # Validate that receipt data is a ReceiptModel if it looks like receipt data
        if data and isinstance(data, dict) and 'items' in data:
            try:
                # Try to validate as ReceiptModel
                validated_data = ReceiptModel(**data)
                return cls(status="success", data=validated_data)
            except Exception as e:
                # If validation fails, return failure with validation error
                return cls(status="failed", error=f"Data validation failed: {e}")

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
