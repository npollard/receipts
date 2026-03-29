"""Data models for receipt processing"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, field_validator, ValidationInfo


class ReceiptItem(BaseModel):
    """Individual receipt item model"""
    name: str
    category: str
    price: Decimal

    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v < 0:
            raise ValueError('Price must be non-negative')
        return v


class Receipt(BaseModel):
    """Main receipt data model"""
    date: Optional[str] = None
    items: List[ReceiptItem] = []
    total: Optional[Decimal] = None

    @field_validator('total')
    @classmethod
    def validate_total(cls, v, info: ValidationInfo):
        # Get the validated data from context
        data = info.data if hasattr(info, 'data') else {}

        if v is not None and 'items' in data:
            calculated_total = sum(item.price for item in data['items'])
            if abs(v - calculated_total) > Decimal('0.01'):  # Allow small rounding differences
                raise ValueError(f'Total {v} does not match sum of items {calculated_total}')
        return v
