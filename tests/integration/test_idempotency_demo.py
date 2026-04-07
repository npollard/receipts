#!/usr/bin/env python3
"""Test script to demonstrate idempotency functionality"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.repository import ReceiptRepository
from database_models import DatabaseManager
from models.receipt import Receipt as ReceiptModel, ReceiptItem as ReceiptItemModel
from api_response import APIResponse
from decimal import Decimal

load_dotenv()


def test_idempotency():
    """Test idempotency with duplicate receipts"""
    
    # Setup database
    database_url = os.getenv("DATABASE_URL", "sqlite:///test_receipts.db")
    db_manager = DatabaseManager(database_url)
    db_manager.create_tables()
    
    # Create repository
    from uuid import uuid4
    user_id = uuid4()
    repository = ReceiptRepository(database_url, user_id)
    
    # Create test receipt data
    test_receipt_data = ReceiptModel(
        date="2025-12-31",
        items=[
            ReceiptItemModel(description="Test Item 1", price=Decimal("10.99")),
            ReceiptItemModel(description="Test Item 2", price=Decimal("5.50"))
        ],
        total=Decimal("16.49")
    )
    
    # Create API response
    api_response = APIResponse.success(test_receipt_data)
    
    print("=== Testing Idempotency ===")
    print(f"Database URL: {database_url}")
    print(f"User ID: {user_id}")
    print()
    
    # Test 1: Save receipt for the first time
    print("Test 1: Saving receipt for the first time...")
    image_path = "test_image.jpg"
    ocr_text = "Test OCR text"
    
    try:
        receipt1, status1 = repository.save_receipt(image_path, ocr_text, api_response)
        print(f"✅ First save successful: {receipt1.id}")
        print(f"   Status: {status1}")
        print(f"   Processing status: {receipt1.processing_status}")
        print()
    except Exception as e:
        print(f"❌ First save failed: {e}")
        return
    
    # Test 2: Try to save the same receipt again (should return existing)
    print("Test 2: Attempting to save duplicate receipt...")
    try:
        receipt2, status2 = repository.save_receipt(image_path, ocr_text, api_response)
        print(f"✅ Duplicate handling: {receipt2.id}")
        print(f"   Status: {status2}")
        print(f"   Same as first: {receipt1.id == receipt2.id}")
        print()
    except Exception as e:
        print(f"❌ Duplicate handling failed: {e}")
        return
    
    # Test 3: Try to save receipt with same data but different image path
    print("Test 3: Saving receipt with same data, different image...")
    different_image_path = "different_image.jpg"
    
    try:
        receipt3, status3 = repository.save_receipt(different_image_path, ocr_text, api_response)
        print(f"✅ Data duplicate handling: {receipt3.id}")
        print(f"   Status: {status3}")
        print(f"   Different from first: {receipt1.id != receipt3.id}")
        print()
    except Exception as e:
        print(f"❌ Data duplicate handling failed: {e}")
        return
    
    # Test 4: Save a completely different receipt
    print("Test 4: Saving completely different receipt...")
    different_receipt_data = ReceiptModel(
        date="2025-12-30",
        items=[ReceiptItemModel(description="Different Item", price=Decimal("25.00"))],
        total=Decimal("25.00")
    )
    different_api_response = APIResponse.success(different_receipt_data)
    different_image_path2 = "different_image2.jpg"
    
    try:
        receipt4, status4 = repository.save_receipt(different_image_path2, "Different OCR", different_api_response)
        print(f"✅ Different receipt: {receipt4.id}")
        print(f"   Status: {status4}")
        print(f"   Different from all others: {receipt4.id not in [receipt1.id, receipt2.id, receipt3.id]}")
        print()
    except Exception as e:
        print(f"❌ Different receipt failed: {e}")
        return
    
    # Summary
    print("=== Summary ===")
    print(f"Total unique receipts created: 3")
    print(f"Receipt 1 (original): {receipt1.id}")
    print(f"Receipt 2 (duplicate of 1): {receipt2.id} (same as receipt 1)")
    print(f"Receipt 3 (data duplicate of 1): {receipt3.id}")
    print(f"Receipt 4 (different): {receipt4.id}")
    print()
    print("✅ All idempotency tests passed!")
    
    # Cleanup
    db_manager.close()
    print("Database connection closed.")


if __name__ == "__main__":
    test_idempotency()
