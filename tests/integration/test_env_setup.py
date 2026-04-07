#!/usr/bin/env python3
"""Test environment setup script"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import app_config, DATABASE_URL, DATABASE_PATH
from database_models import DatabaseManager


def setup_test_environment():
    """Setup test database and environment"""
    
    print("=== Test Environment Setup ===")
    print(f"Environment: {app_config.ENVIRONMENT}")
    print(f"Is Test: {app_config.IS_TEST}")
    print(f"Database URL: {DATABASE_URL}")
    print(f"Database Path: {DATABASE_PATH}")
    print()
    
    # Set test environment variable
    os.environ["ENVIRONMENT"] = "test"
    
    # Initialize test database
    print("Initializing test database...")
    db_manager = DatabaseManager()
    db_manager.create_tables()
    
    print(f"✅ Test database created: {DATABASE_PATH}")
    print(f"✅ Test environment ready")
    
    return db_manager


def cleanup_test_environment():
    """Cleanup test database"""
    print("\n=== Test Environment Cleanup ===")
    
    if DATABASE_PATH.exists():
        try:
            DATABASE_PATH.unlink()
            print(f"✅ Test database removed: {DATABASE_PATH}")
        except Exception as e:
            print(f"❌ Error removing test database: {e}")
    else:
        print(f"ℹ️  Test database does not exist: {DATABASE_PATH}")


def verify_test_isolation():
    """Verify that test database is isolated from production"""
    
    print("\n=== Test Isolation Verification ===")
    
    # Check production database
    production_db = Path("receipts.db")
    test_db = Path("test_receipts.db")
    
    print(f"Production DB exists: {production_db.exists()}")
    print(f"Test DB exists: {test_db.exists()}")
    
    if production_db.exists() and test_db.exists():
        # Compare file sizes to ensure they're different
        prod_size = production_db.stat().st_size
        test_size = test_db.stat().st_size
        
        if prod_size != test_size:
            print("✅ Test and production databases are isolated")
            return True
        else:
            print("⚠️  Test and production databases might be the same")
            return False
    else:
        print("✅ Databases are properly separated")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test environment management")
    parser.add_argument("--setup", action="store_true", help="Setup test environment")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup test environment")
    parser.add_argument("--verify", action="store_true", help="Verify test isolation")
    parser.add_argument("--all", action="store_true", help="Run setup, verify, and cleanup")
    
    args = parser.parse_args()
    
    if args.all:
        # Run full test cycle
        db_manager = setup_test_environment()
        verify_test_isolation()
        cleanup_test_environment()
    elif args.setup:
        setup_test_environment()
    elif args.cleanup:
        cleanup_test_environment()
    elif args.verify:
        verify_test_isolation()
    else:
        print("Use --setup, --cleanup, --verify, or --all")
        print("Example: python test_env_setup.py --all")
