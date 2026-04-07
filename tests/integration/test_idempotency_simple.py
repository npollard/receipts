"""Test receipt data idempotency functionality - hashing only"""

import json
from database_models import calculate_receipt_data_hash

def test_receipt_data_hashing():
    """Test receipt data hash generation and consistency"""
    
    print("🧪 Testing Receipt Data Hashing")
    print("=" * 40)
    
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
    
    # Edge cases
    receipt_data_4 = {
        "date": None,
        "total": None,
        "merchant": "",
        "items": []
    }
    
    receipt_data_5 = {}
    
    # Calculate hashes
    hash_1 = calculate_receipt_data_hash(receipt_data_1)
    hash_2 = calculate_receipt_data_hash(receipt_data_2)
    hash_3 = calculate_receipt_data_hash(receipt_data_3)
    hash_4 = calculate_receipt_data_hash(receipt_data_4)
    hash_5 = calculate_receipt_data_hash(receipt_data_5)
    
    print(f"\n📊 Hash Results:")
    print(f"   Hash 1 (original):  {hash_1}")
    print(f"   Hash 2 (reordered): {hash_2}")
    print(f"   Hash 3 (different): {hash_3}")
    print(f"   Hash 4 (empty):     {hash_4}")
    print(f"   Hash 5 (minimal):   {hash_5}")
    
    # Test idempotency
    print(f"\n✅ Idempotency Tests:")
    
    # Test 1: Same data should produce same hash
    if hash_1 == hash_2:
        print("   ✅ Same receipt data produces same hash regardless of formatting")
    else:
        print("   ❌ Same receipt data should produce same hash")
        return False
    
    # Test 2: Different data should produce different hash
    if hash_1 != hash_3:
        print("   ✅ Different receipt data produces different hash")
    else:
        print("   ❌ Different receipt data should produce different hash")
        return False
    
    # Test 3: Empty data should be consistent
    if hash_4 == hash_5:
        print("   ✅ Empty/missing data handled consistently")
    else:
        print("   ❌ Empty/missing data should be handled consistently")
        return False
    
    # Test 4: Hash length should be consistent (SHA-256 = 64 chars)
    if len(hash_1) == 64 and len(hash_2) == 64 and len(hash_3) == 64:
        print("   ✅ Hash length consistent (SHA-256 = 64 characters)")
    else:
        print("   ❌ Hash length should be 64 characters (SHA-256)")
        return False
    
    # Test 5: Hash should be deterministic
    hash_1_again = calculate_receipt_data_hash(receipt_data_1)
    if hash_1 == hash_1_again:
        print("   ✅ Hash generation is deterministic")
    else:
        print("   ❌ Hash generation should be deterministic")
        return False
    
    print(f"\n🎉 All idempotency tests passed!")
    return True

def demonstrate_idempotency_workflow():
    """Demonstrate how idempotency works in the workflow"""
    
    print(f"\n🔄 Idempotency Workflow Demonstration")
    print("=" * 45)
    
    # Simulate processing the same receipt multiple times
    receipt_data = {
        "date": "2024-03-15",
        "total": 42.75,
        "merchant": "Target",
        "items": [
            {"description": "Coffee", "price": 8.99, "quantity": 2},
            {"description": "Sandwich", "price": 6.99, "quantity": 1},
            {"description": "Chips", "price": 3.99, "quantity": 1},
            {"description": "Soda", "price": 2.49, "quantity": 2}
        ]
    }
    
    print(f"\n📝 Processing receipt: {receipt_data['merchant']} - ${receipt_data['total']}")
    
    # Calculate hash
    receipt_hash = calculate_receipt_data_hash(receipt_data)
    print(f"🔑 Receipt data hash: {receipt_hash}")
    
    # Simulate multiple processing attempts
    attempts = [
        {"attempt": 1, "image": "receipt_1.jpg", "result": "success"},
        {"attempt": 2, "image": "receipt_2.jpg", "result": "duplicate_detected"},
        {"attempt": 3, "image": "receipt_3.jpg", "result": "duplicate_detected"}
    ]
    
    print(f"\n📋 Processing Attempts:")
    for attempt in attempts:
        print(f"   Attempt {attempt['attempt']}: {attempt['image']} -> {attempt['result']}")
        
        if attempt['result'] == 'duplicate_detected':
            print(f"      💡 Skipping: Receipt with hash {receipt_hash[:12]}... already exists")
        else:
            print(f"      ✅ Processed: New receipt stored with hash {receipt_hash[:12]}...")
    
    print(f"\n💰 Benefits:")
    print(f"   • Avoids duplicate processing costs")
    print(f"   • Maintains data consistency") 
    print(f"   • Reduces database storage needs")
    print(f"   • Improves processing speed for duplicates")

if __name__ == "__main__":
    # Test the hashing functionality
    success = test_receipt_data_hashing()
    
    if success:
        # Demonstrate the workflow
        demonstrate_idempotency_workflow()
        
        print(f"\n✨ Summary:")
        print(f"   • Receipt data hashing provides robust idempotency")
        print(f"   • Normalizes data to handle formatting differences")
        print(f"   • Uses SHA-256 for consistent, collision-resistant hashes")
        print(f"   • Ready for integration with database persistence")
    else:
        print(f"\n❌ Tests failed - please check the implementation")
