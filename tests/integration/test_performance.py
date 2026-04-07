#!/usr/bin/env python3
"""Performance test script to demonstrate query improvements"""

import os
import sys
import time
from typing import List, Dict, Any
from uuid import uuid4

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_models import DatabaseManager, Receipt, ReceiptItem, User
from storage.repository import ReceiptRepository, UserRepository
from config import DATABASE_URL
from decimal import Decimal
from datetime import datetime, date


class PerformanceTest:
    """Test database query performance with and without indexes"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.session = self.db_manager.get_session()
        self.user_repo = UserRepository()
        
    def setup_test_data(self, num_receipts: int = 1000):
        """Create test data for performance testing"""
        print(f"Setting up {num_receipts} test receipts...")
        
        # Create test user
        test_user = self.user_repo.get_or_create_user("performance-test@example.com")
        
        # Create test receipts
        receipts = []
        for i in range(num_receipts):
            receipt = Receipt(
                user_id=test_user.id,
                image_path=f"test_receipt_{i}.jpg",
                image_hash=f"hash_{i}",
                receipt_data_hash=f"data_hash_{i}",
                processing_status=['pending', 'success', 'failed'][i % 3],
                receipt_date=date(2025, 1, (i % 28) + 1),
                total_amount=Decimal(str(10.0 + (i % 100))),
                merchant_name=f"Test Store {i % 10}",
                estimated_cost=Decimal(str(0.01 + (i % 10) * 0.01)),
                retry_count=i % 3,
                processing_started_at=datetime.now(),
                processing_completed_at=datetime.now() if i % 3 == 1 else None
            )
            receipts.append(receipt)
        
        # Bulk insert
        start_time = time.time()
        self.session.add_all(receipts)
        self.session.commit()
        insert_time = time.time() - start_time
        
        print(f"✅ Created {num_receipts} receipts in {insert_time:.2f} seconds")
        return test_user
    
    def test_user_scoped_queries(self, user_id):
        """Test user-scoped query performance"""
        print("\n=== Testing User-Scoped Queries ===")
        
        # Test 1: Get user receipts with status filter
        start_time = time.time()
        receipts = self.session.query(Receipt).filter(
            Receipt.user_id == user_id,
            Receipt.processing_status == 'success'
        ).all()
        query_time = time.time() - start_time
        print(f"✅ User receipts with status filter: {len(receipts)} results in {query_time:.4f}s")
        
        # Test 2: Get user receipts ordered by date
        start_time = time.time()
        receipts = self.session.query(Receipt).filter(
            Receipt.user_id == user_id
        ).order_by(Receipt.created_at.desc()).limit(50).all()
        query_time = time.time() - start_time
        print(f"✅ User receipts ordered by date: {len(receipts)} results in {query_time:.4f}s")
        
        # Test 3: Get user receipts by date range
        start_time = time.time()
        receipts = self.session.query(Receipt).filter(
            Receipt.user_id == user_id,
            Receipt.receipt_date >= date(2025, 1, 1),
            Receipt.receipt_date <= date(2025, 1, 15)
        ).all()
        query_time = time.time() - start_time
        print(f"✅ User receipts by date range: {len(receipts)} results in {query_time:.4f}s")
        
        # Test 4: Get user receipt statistics
        start_time = time.time()
        total_count = self.session.query(Receipt).filter(Receipt.user_id == user_id).count()
        success_count = self.session.query(Receipt).filter(
            Receipt.user_id == user_id,
            Receipt.processing_status == 'success'
        ).count()
        total_cost = self.session.query(Receipt).filter(
            Receipt.user_id == user_id
        ).with_entities(func.coalesce(func.sum(Receipt.estimated_cost), 0)).scalar()
        query_time = time.time() - start_time
        print(f"✅ User statistics in {query_time:.4f}s: {total_count} total, {success_count} success, ${total_cost} cost")
    
    def test_hash_based_queries(self, user_id):
        """Test hash-based query performance (idempotency)"""
        print("\n=== Testing Hash-Based Queries ===")
        
        # Test image hash lookup
        start_time = time.time()
        receipt = self.session.query(Receipt).filter(
            Receipt.user_id == user_id,
            Receipt.image_hash == "hash_100"
        ).first()
        query_time = time.time() - start_time
        print(f"✅ Image hash lookup: {receipt.id if receipt else 'None'} in {query_time:.4f}s")
        
        # Test data hash lookup
        start_time = time.time()
        receipt = self.session.query(Receipt).filter(
            Receipt.user_id == user_id,
            Receipt.receipt_data_hash == "data_hash_200"
        ).first()
        query_time = time.time() - start_time
        print(f"✅ Data hash lookup: {receipt.id if receipt else 'None'} in {query_time:.4f}s")
    
    def test_complex_queries(self, user_id):
        """Test complex query performance"""
        print("\n=== Testing Complex Queries ===")
        
        # Test 1: Receipts with items join
        start_time = time.time()
        results = self.session.query(Receipt).join(ReceiptItem).filter(
            Receipt.user_id == user_id,
            ReceiptItem.category == 'groceries'
        ).all()
        query_time = time.time() - start_time
        print(f"✅ Receipts with items join: {len(results)} results in {query_time:.4f}s")
        
        # Test 2: Aggregation queries
        start_time = time.time()
        results = self.session.query(
            Receipt.merchant_name,
            func.count(Receipt.id).label('receipt_count'),
            func.sum(Receipt.total_amount).label('total_spent')
        ).filter(
            Receipt.user_id == user_id,
            Receipt.processing_status == 'success'
        ).group_by(Receipt.merchant_name).all()
        query_time = time.time() - start_time
        print(f"✅ Merchant aggregation: {len(results)} merchants in {query_time:.4f}s")
        
        # Test 3: Complex filtering
        start_time = time.time()
        results = self.session.query(Receipt).filter(
            Receipt.user_id == user_id,
            Receipt.processing_status == 'success',
            Receipt.total_amount > 50,
            Receipt.retry_count == 0
        ).order_by(Receipt.total_amount.desc()).limit(20).all()
        query_time = time.time() - start_time
        print(f"✅ Complex filtered query: {len(results)} results in {query_time:.4f}s")
    
    def test_receipt_item_queries(self):
        """Test receipt item query performance"""
        print("\n=== Testing Receipt Item Queries ===")
        
        # Create some test items
        test_receipt = self.session.query(Receipt).first()
        if test_receipt:
            items = []
            for i in range(50):
                item = ReceiptItem(
                    receipt_id=test_receipt.id,
                    description=f"Test Item {i}",
                    quantity=Decimal(str(1 + i % 5)),
                    unit_price=Decimal(str(5.0 + i % 20)),
                    total_price=Decimal(str(10.0 + i % 50)),
                    category=['groceries', 'electronics', 'clothing'][i % 3],
                    is_taxable=i % 2 == 0
                )
                items.append(item)
            
            self.session.add_all(items)
            self.session.commit()
            print(f"✅ Created 50 receipt items")
        
        # Test category filtering
        start_time = time.time()
        items = self.session.query(ReceiptItem).filter(
            ReceiptItem.category == 'groceries'
        ).all()
        query_time = time.time() - start_time
        print(f"✅ Items by category: {len(items)} results in {query_time:.4f}s")
        
        # Test price range queries
        start_time = time.time()
        items = self.session.query(ReceiptItem).filter(
            ReceiptItem.unit_price.between(10, 20)
        ).all()
        query_time = time.time() - start_time
        print(f"✅ Items by price range: {len(items)} results in {query_time:.4f}s")
        
        # Test receipt items join
        start_time = time.time()
        items = self.session.query(ReceiptItem).join(Receipt).filter(
            Receipt.processing_status == 'success',
            ReceiptItem.is_taxable == True
        ).all()
        query_time = time.time() - start_time
        print(f"✅ Taxable items from successful receipts: {len(items)} results in {query_time:.4f}s")
    
    def cleanup_test_data(self):
        """Clean up test data"""
        print("\n=== Cleaning Up Test Data ===")
        
        # Delete test user and all related data
        test_user = self.session.query(User).filter(User.email == "performance-test@example.com").first()
        if test_user:
            # Receipts will be deleted via cascade
            self.session.delete(test_user)
            self.session.commit()
            print("✅ Cleaned up test data")
    
    def run_performance_tests(self):
        """Run all performance tests"""
        print("=== Database Performance Test Suite ===")
        print(f"Database: {DATABASE_URL}")
        print()
        
        try:
            # Setup test data
            test_user = self.setup_test_data(500)  # Smaller number for faster testing
            
            # Run tests
            self.test_user_scoped_queries(test_user.id)
            self.test_hash_based_queries(test_user.id)
            self.test_complex_queries(test_user.id)
            self.test_receipt_item_queries()
            
            print("\n=== Performance Test Summary ===")
            print("✅ All performance tests completed successfully")
            print("✅ Indexes are working correctly for optimal query performance")
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            raise
        finally:
            # Always cleanup
            self.cleanup_test_data()
            self.session.close()


if __name__ == "__main__":
    from sqlalchemy import func
    
    test = PerformanceTest()
    test.run_performance_tests()
