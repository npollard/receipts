"""Test receipt data idempotency functionality"""

import os
import json
from uuid import uuid4
from dotenv import load_dotenv

from database_models import DatabaseManager
from core.hashing import calculate_receipt_data_hash
from receipt_persistence import ReceiptPersistence

# Load environment variables
load_dotenv()

def test_receipt_data_hashing():
    """Test receipt data hash generation and consistency"""

    # Sample receipt data
    receipt_data_1 = {
        "date": "2024-03-15",
        "total": 25.50,
        "merchant": "Grocery Store",
        "items": [
            {"description": "Milk", "price": 4.50, "quantity": 1},
            {"description": "Bread", "price": 3.00, "quantity": 2},
            {"description": "Eggs", "price": 5.50, "quantity": 1}
        ]
    }

    # Same data with different formatting and item order
    receipt_data_2 = {
        "merchant": "Grocery Store",
        "date": "2024-03-15",
        "total": 25.50,
        "items": [
            {"quantity": 1, "description": "milk", "price": 4.50},
            {"quantity": 2, "description": "bread", "price": 3.00},
            {"quantity": 1, "description": "EGGS", "price": 5.50}
        ]
    }

    # Different data
    receipt_data_3 = {
        "date": "2024-03-15",
        "total": 30.00,
        "merchant": "Grocery Store",
        "items": [
            {"description": "Milk", "price": 4.50, "quantity": 1},
            {"description": "Bread", "price": 3.00, "quantity": 2},
            {"description": "Cheese", "price": 8.00, "quantity": 1}
        ]
    }

    # Calculate hashes
    hash_1 = calculate_receipt_data_hash(receipt_data_1)
    hash_2 = calculate_receipt_data_hash(receipt_data_2)
    hash_3 = calculate_receipt_data_hash(receipt_data_3)

    print(f"Hash 1: {hash_1}")
    print(f"Hash 2: {hash_2}")
    print(f"Hash 3: {hash_3}")

    # Test idempotency
    assert hash_1 == hash_2, "Same receipt data should produce same hash regardless of formatting"
    assert hash_1 != hash_3, "Different receipt data should produce different hash"

    print("✅ Receipt data hashing works correctly!")


def test_database_idempotency():
    """Test database-level idempotency with receipt data"""

    # Database configuration (use unique temp file per test to avoid locking issues)
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    DATABASE_URL = f"sqlite:///{temp_db.name}"

    # Initialize database manager
    db_manager = DatabaseManager(DATABASE_URL)
    db_manager.create_tables()

    # Create test user
    user_id = uuid4()
    persistence = ReceiptPersistence(db_manager, user_id)
    user = persistence.get_or_create_user("test@example.com")

    # Sample receipt data
    receipt_data = {
        "date": "2024-03-15",
        "total": 25.50,
        "merchant": "Test Store",
        "items": [
            {"description": "Item 1", "price": 10.00, "quantity": 1},
            {"description": "Item 2", "price": 15.50, "quantity": 1}
        ]
    }

    try:
        # First attempt - should create new receipt
        print("📝 First processing attempt...")
        duplicate_1 = persistence.check_duplicate_receipt_data(receipt_data)
        assert duplicate_1 is None, "First attempt should not find duplicate"
        print("✅ No duplicate found (as expected)")

        # Create a mock receipt record to simulate successful processing
        from api_response import APIResponse
        from datetime import datetime

        # Create pending receipt
        receipt_1 = persistence.create_pending_receipt("test_image_1.jpg", "OCR text here")

        # Update with success (this will set the data hash)
        success_response = APIResponse.success(receipt_data)
        updated_receipt_1 = persistence.update_receipt_success(
            receipt_1.id, success_response, 100, 50, 0.000150
        )

        print(f"✅ Created receipt: {updated_receipt_1.id}")
        print(f"   Data hash: {updated_receipt_1.receipt_data_hash}")

        # Second attempt with same data - should find duplicate
        print("\n📝 Second processing attempt with same data...")
        duplicate_2 = persistence.check_duplicate_receipt_data(receipt_data)
        assert duplicate_2 is not None, "Second attempt should find duplicate"
        assert duplicate_2.id == updated_receipt_1.id, "Should find the same receipt"
        print(f"✅ Found duplicate receipt: {duplicate_2.id}")

        # Third attempt with different data - should not find duplicate
        print("\n📝 Third processing attempt with different data...")
        different_data = receipt_data.copy()
        different_data["total"] = 30.00
        different_data["items"][0]["price"] = 14.50

        duplicate_3 = persistence.check_duplicate_receipt_data(different_data)
        assert duplicate_3 is None, "Different data should not find duplicate"
        print("✅ No duplicate found for different data (as expected)")

        print("\n🎉 Database idempotency test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

    finally:
        # Cleanup
        db_manager.close()
        # Remove test database file if using SQLite
        if DATABASE_URL.startswith("sqlite:///"):
            db_file = DATABASE_URL.replace("sqlite:///", "")
            if os.path.exists(db_file):
                os.remove(db_file)

if __name__ == "__main__":
    print("🧪 Testing Receipt Data Idempotency")
    print("=" * 50)

    # Test 1: Hash consistency
    print("\n1. Testing hash consistency...")
    test_receipt_data_hashing()

    # Test 2: Database idempotency
    print("\n2. Testing database idempotency...")
    test_database_idempotency()

    print("\n✨ All tests completed!")
