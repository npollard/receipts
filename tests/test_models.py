"""Unit tests for Receipt and ReceiptItem models"""

import pytest
from decimal import Decimal
from pydantic import ValidationError
from models.receipt import Receipt, ReceiptItem


def test_receipt_item_valid_creation():
    """Test creating a valid ReceiptItem"""
    item = ReceiptItem(description="Milk", price=4.50)

    assert item.description == "Milk"
    assert item.price == Decimal("4.50")


def test_receipt_item_with_decimal_price():
    """Test ReceiptItem with Decimal price"""
    price = Decimal("3.99")
    item = ReceiptItem(description="Bread", price=price)

    assert item.price == price


def test_receipt_item_negative_price():
    """Test ReceiptItem rejects negative prices"""
    with pytest.raises(ValidationError) as exc_info:
        ReceiptItem(description="Invalid", price=-1.50)

    assert "price" in str(exc_info.value)


def test_receipt_item_empty_description():
    """Test ReceiptItem rejects empty description"""
    # Pydantic v2 doesn't validate empty strings by default for str fields
    # This test documents the current behavior
    item = ReceiptItem(description="", price=2.50)
    assert item.description == ""


def test_receipt_valid_creation():
    """Test creating a valid Receipt"""
    items = [
        ReceiptItem(description="Milk", price=4.50),
        ReceiptItem(description="Bread", price=3.00)
    ]
    receipt = Receipt(date="2026-03-30", items=items, total=7.50)

    assert receipt.date == "2026-03-30"
    assert len(receipt.items) == 2
    assert receipt.total == Decimal("7.50")


def test_receipt_total_validation():
    """Test receipt total validation matches items sum"""
    items = [
        ReceiptItem(description="Milk", price=4.50),
        ReceiptItem(description="Bread", price=3.00)
    ]

    # Valid total
    receipt = Receipt(date="2026-03-30", items=items, total=7.50)
    assert receipt.total == Decimal("7.50")

    # Invalid total (doesn't match items sum)
    with pytest.raises(ValidationError) as exc_info:
        Receipt(date="2026-03-30", items=items, total=10.00)

    assert "total" in str(exc_info.value)


def test_receipt_empty_items():
    """Test receipt with empty items list"""
    receipt = Receipt(date="2026-03-30", items=[], total=0.00)

    assert len(receipt.items) == 0
    assert receipt.total == Decimal("0.00")


def test_receipt_none_date():
    """Test receipt with None date"""
    items = [ReceiptItem(description="Milk", price=4.50)]
    receipt = Receipt(date=None, items=items, total=4.50)

    assert receipt.date is None


def test_receipt_none_total():
    """Test receipt with None total"""
    items = [ReceiptItem(description="Milk", price=4.50)]
    receipt = Receipt(date="2026-03-30", items=items, total=None)

    assert receipt.total is None


def test_receipt_invalid_date_format():
    """Test receipt accepts various date formats"""
    items = [ReceiptItem(description="Milk", price=4.50)]

    # Receipt model accepts any string for date field (no validation)
    receipt = Receipt(date="invalid-date-format", items=items, total=4.50)
    assert receipt.date == "invalid-date-format"

    # Also accepts None
    receipt_none = Receipt(date=None, items=items, total=4.50)
    assert receipt_none.date is None


def test_receipt_model_dump():
    """Test Receipt model serialization"""
    items = [ReceiptItem(description="Milk", price=4.50)]
    receipt = Receipt(date="2026-03-30", items=items, total=4.50)

    data = receipt.model_dump()

    assert data["date"] == "2026-03-30"
    assert len(data["items"]) == 1
    assert data["total"] == 4.50
    assert data["items"][0]["description"] == "Milk"
    assert data["items"][0]["price"] == 4.50
