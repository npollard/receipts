"""Fluent builder for creating test receipts."""

from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
from uuid import uuid4


class ReceiptBuilder:
    """Fluent builder for constructing test receipt data.
    
    Example:
        >>> receipt = (ReceiptBuilder()
        ...     .with_merchant("Grocery Store")
        ...     .with_total(25.50)
        ...     .with_date("2024-01-15")
        ...     .with_item("Milk", 3.99)
        ...     .with_item("Bread", 2.50)
        ...     .build())
    """
    
    def __init__(self):
        self._data = {
            "id": str(uuid4()),
            "merchant_name": "Test Store",
            "total_amount": 0.0,
            "receipt_date": "2024-01-01",
            "items": [],
            "currency": "USD",
            "tax_amount": None,
            "image_path": "test_receipt.jpg",
            "image_hash": f"hash_{uuid4().hex[:8]}",
            "receipt_data_hash": f"data_{uuid4().hex[:8]}",
            "status": "success",
            "user_id": "default_user",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
    
    def with_id(self, receipt_id: str) -> "ReceiptBuilder":
        """Set receipt ID."""
        self._data["id"] = receipt_id
        return self
    
    def with_merchant(self, name: str) -> "ReceiptBuilder":
        """Set merchant name."""
        self._data["merchant_name"] = name
        return self
    
    def with_total(self, amount: float) -> "ReceiptBuilder":
        """Set total amount."""
        self._data["total_amount"] = float(amount)
        return self
    
    def with_date(self, date: str) -> "ReceiptBuilder":
        """Set receipt date (ISO format string)."""
        self._data["receipt_date"] = date
        return self
    
    def with_currency(self, currency: str) -> "ReceiptBuilder":
        """Set currency code."""
        self._data["currency"] = currency
        return self
    
    def with_tax(self, amount: float) -> "ReceiptBuilder":
        """Set tax amount."""
        self._data["tax_amount"] = float(amount)
        return self
    
    def with_item(
        self,
        description: str,
        price: float,
        quantity: int = 1
    ) -> "ReceiptBuilder":
        """Add an item to the receipt."""
        item = {
            "description": description,
            "price": float(price),
            "quantity": quantity,
        }
        self._data["items"].append(item)
        return self
    
    def with_items(self, items: List[Dict[str, Any]]) -> "ReceiptBuilder":
        """Set all items at once."""
        self._data["items"] = list(items)
        return self
    
    def with_image_path(self, path: str) -> "ReceiptBuilder":
        """Set image path."""
        self._data["image_path"] = path
        return self
    
    def with_image_hash(self, hash_value: str) -> "ReceiptBuilder":
        """Set image hash (for idempotency testing)."""
        self._data["image_hash"] = hash_value
        return self
    
    def with_data_hash(self, hash_value: str) -> "ReceiptBuilder":
        """Set receipt data hash (for idempotency testing)."""
        self._data["receipt_data_hash"] = hash_value
        return self
    
    def with_user_id(self, user_id: str) -> "ReceiptBuilder":
        """Set user ID."""
        self._data["user_id"] = user_id
        return self
    
    def with_status(self, status: str) -> "ReceiptBuilder":
        """Set processing status."""
        self._data["status"] = status
        return self
    
    def as_pending(self) -> "ReceiptBuilder":
        """Set status to pending."""
        self._data["status"] = "pending"
        return self
    
    def as_failed(self) -> "ReceiptBuilder":
        """Set status to failed."""
        self._data["status"] = "failed"
        return self
    
    def as_success(self) -> "ReceiptBuilder":
        """Set status to success."""
        self._data["status"] = "success"
        return self
    
    def build(self) -> Any:
        """Build and return the receipt DTO."""
        # Create a simple object with attributes
        class SimpleReceiptDTO:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        return SimpleReceiptDTO(self._data)
    
    def build_dict(self) -> Dict[str, Any]:
        """Build and return as dictionary."""
        return dict(self._data)
    
    @classmethod
    def grocery_store(cls, total: float = 25.50) -> "ReceiptBuilder":
        """Create a pre-configured grocery store receipt."""
        return (cls()
            .with_merchant("Grocery Store")
            .with_total(total)
            .with_item("Milk", 3.99)
            .with_item("Bread", 2.50)
            .with_item("Eggs", 4.99))
    
    @classmethod
    def restaurant(cls, total: float = 45.00) -> "ReceiptBuilder":
        """Create a pre-configured restaurant receipt."""
        return (cls()
            .with_merchant("Bistro Restaurant")
            .with_total(total)
            .with_item("Appetizer", 12.00)
            .with_item("Main Course", 28.00)
            .with_tax(total * 0.1))
    
    @classmethod
    def gas_station(cls, total: float = 35.00) -> "ReceiptBuilder":
        """Create a pre-configured gas station receipt."""
        return (cls()
            .with_merchant("Shell Gas Station")
            .with_total(total)
            .with_item("Premium Gas", total))
