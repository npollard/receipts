#!/usr/bin/env python3

import sys
import os

def test_consolidated_orchestration():
    """Test the consolidated orchestration flow"""
    print("Testing consolidated orchestration flow...")

    # Test 1: Basic ReceiptProcessor initialization
    print("\n=== TEST 1: BASIC INITIALIZATION ===")
    try:
        from pipeline.processor import ReceiptProcessor
        processor = ReceiptProcessor()
        print("ReceiptProcessor initialized successfully without database")

        # Check core components
        assert hasattr(processor, 'image_processor')
        assert hasattr(processor, 'ai_parser')
        assert hasattr(processor, 'token_usage')
        print("Core components present")

        # Check methods
        required_methods = [
            'process_image', 'process_directly', 'get_token_usage_summary',
            'reset_token_usage', 'get_current_user', 'switch_user'
        ]
        for method in required_methods:
            assert hasattr(processor, method), f"Missing method: {method}"
        print("All required methods present")

    except Exception as e:
        print(f"Initialization failed: {e}")
        assert False, f"Initialization failed: {e}"

    # Test 2: ReceiptProcessor with database
    print("\n=== TEST 2: DATABASE INITIALIZATION ===")
    try:
        from infrastructure.database import DatabaseManager
        db_manager = DatabaseManager()
        # Create tables for test
        db_manager.create_tables()
        processor_db = ReceiptProcessor(db_manager=db_manager)
        print("ReceiptProcessor initialized successfully with database")

        # Check database-specific components
        assert hasattr(processor_db, 'user_manager')
        assert hasattr(processor_db, 'repository')
        assert hasattr(processor_db, 'persistence')
        print("Database components present")

    except Exception as e:
        print(f"Database initialization failed: {e}")
        assert False, f"Database initialization failed: {e}"

    # Test 3: Method availability and basic functionality
    print("\n=== TEST 3: METHOD FUNCTIONALITY ===")
    try:
        processor = ReceiptProcessor()

        # Test token usage summary
        summary = processor.get_token_usage_summary()
        assert isinstance(summary, str)
        print("Token usage summary works")

        # Test token usage reset
        processor.reset_token_usage()
        assert processor.token_usage.get_total_tokens() == 0
        print("Token usage reset works")

        # Test user context (no database mode)
        context = processor.get_user_context()
        assert isinstance(context, dict)
        assert 'user_id' in context
        assert 'email' in context
        print("User context works")

    except Exception as e:
        print(f"Method functionality test failed: {e}")
        assert False, f"Method functionality test failed: {e}"

    # Test 4: Import compatibility
    print("\n=== TEST 4: IMPORT COMPATIBILITY ===")
    try:
        # Test the imports that main.py uses
        from pipeline.processor import (
            ReceiptProcessor,
            validate_and_get_image_files,
            print_batch_summary,
            print_token_usage_summary,
            print_usage_summary
        )
        print("All main.py imports work correctly")

        # Note: process_batch_images removed (use BatchProcessingService directly)
        # Note: save_token_usage_to_persistence removed (persistence moved to application layer)

        # Test legacy function
        from pipeline.processor import process_receipt
        print("Legacy function import works")

    except Exception as e:
        print(f"Import compatibility test failed: {e}")
        assert False, f"Import compatibility test failed: {e}"

    # Test 5: Pipeline functions
    print("\n=== TEST 5: PIPELINE FUNCTIONS ===")
    try:
        # Test image validation
        from pathlib import Path
        imgs_dir = Path('imgs')
        image_files = validate_and_get_image_files(imgs_dir)
        print(f"Image validation works (found {len(image_files)} files)")

        # Test batch summary function
        print_batch_summary(5, 2, 7)
        print("Batch summary function works")

        # Test usage summary function
        from tracking import TokenUsage
        token_usage = TokenUsage()
        token_usage.add_usage(100, 50)
        print_token_usage_summary(token_usage)
        print("Token usage summary function works")

    except Exception as e:
        print(f"Pipeline functions test failed: {e}")
        assert False, f"Pipeline functions test failed: {e}"

    # Test 6: Error handling
    print("\n=== TEST 6: ERROR HANDLING ===")
    try:
        processor = ReceiptProcessor()

        # Test processing non-existent image
        result = processor.process_image("non_existent.jpg")
        assert result.status == "failed"
        print("Error handling works for non-existent image")

    except Exception as e:
        print(f"Error handling test failed: {e}")
        assert False, f"Error handling test failed: {e}"

    print("\n=== ALL TESTS PASSED ===")
    print("Consolidated orchestration flow is working correctly!")
    assert True

def test_architecture_flow():
    """Test that the architecture flow is correct"""
    print("\n=== TESTING ARCHITECTURE FLOW ===")

    try:
        # Test main.py flow
        print("Testing main.py flow...")

        # Import what main.py imports
        from pipeline.processor import ReceiptProcessor
        from infrastructure.database import DatabaseManager

        # Initialize like main.py does
        db_manager = DatabaseManager()
        # Create tables for test
        db_manager.create_tables()
        processor = ReceiptProcessor(db_manager=db_manager)

        print("main.py flow works correctly")

        # Test the clear flow: main.py -> pipeline -> services -> domain -> storage
        print("\nTesting architecture layers...")

        # Pipeline layer (we're here)
        assert hasattr(processor, 'process_image')
        print("Pipeline layer: OK")

        # Services layer (processor uses these)
        assert hasattr(processor, 'image_processor')  # OCR service
        assert hasattr(processor, 'ai_parser')        # Parser service
        print("Services layer: OK")

        # Domain layer (services use these)
        from domain.models.receipt import Receipt, ReceiptItem
        print("Domain layer: OK")

        # Storage layer (processor uses these)
        assert hasattr(processor, 'repository')  # Database storage
        print("Storage layer: OK")

        print("Architecture flow: main.py -> pipeline -> services -> domain -> storage")
        print("All layers working correctly!")

    except Exception as e:
        print(f"Architecture flow test failed: {e}")
        assert False, f"Architecture flow test failed: {e}"

if __name__ == '__main__':
    success = test_consolidated_orchestration()
    if success:
        success = test_architecture_flow()

    if success:
        print("\n" + "="*60)
        print("CONSOLIDATED ORCHESTRATION SUCCESSFUL!")
        print("All orchestration layers working correctly.")
        print("Architecture flow: main.py -> pipeline -> services -> domain -> storage")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("CONSOLIDATED ORCHESTRATION FAILED!")
        print("Some orchestration components are not working.")
        print("="*60)
        sys.exit(1)
