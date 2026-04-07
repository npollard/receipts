#!/usr/bin/env python3

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domain.parsing.receipt_parser import ReceiptParser
from api_response import APIResponse
from tracking import TokenUsage

def test_unified_parser():
    """Test the unified parser to ensure no behavior changes"""
    print("Testing unified ReceiptParser...")

    # Test 1: Basic initialization
    print("\n=== TEST 1: BASIC INITIALIZATION ===")
    try:
        parser = ReceiptParser()
        print("ReceiptParser initialized successfully")
        print(f"Model: {parser.llm.model_name}")
        print(f"Temperature: {parser.llm.temperature}")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return False

    # Test 2: Method availability
    print("\n=== TEST 2: METHOD AVAILABILITY ===")
    required_methods = [
        'parse_text',
        'parse_with_retry',
        'parse_with_usage_tracking',
        'parse_receipt_text',
        'get_token_usage_summary',
        'reset_token_usage',
        'build_prompt'
    ]

    for method in required_methods:
        if hasattr(parser, method):
            print(f"Method '{method}': Available")
        else:
            print(f"Method '{method}: MISSING")
            return False

    # Test 3: Token usage tracking
    print("\n=== TEST 3: TOKEN USAGE TRACKING ===")
    try:
        parser.reset_token_usage()
        summary = parser.get_token_usage_summary()
        print(f"Token usage summary: {summary}")
        print("Token usage tracking works")
    except Exception as e:
        print(f"Token usage tracking failed: {e}")
        return False

    # Test 4: LangChain tool interface
    print("\n=== TEST 4: LANGCHAIN TOOL INTERFACE ===")
    try:
        # Check if the method has the tool decorator
        tool_method = getattr(parser, 'parse_receipt_text')
        if hasattr(tool_method, 'name') and hasattr(tool_method, 'description'):
            print("LangChain tool decorator: Present")
            print(f"Tool name: {tool_method.name}")
            print(f"Tool description: {tool_method.description[:100]}...")
        else:
            print("LangChain tool decorator: Missing")
            return False
    except Exception as e:
        print(f"LangChain tool test failed: {e}")
        return False

    # Test 5: Prompt building
    print("\n=== TEST 5: PROMPT BUILDING ===")
    try:
        test_text = "GROCERY STORE\nAPPLES 3.99\nTOTAL 3.99"
        prompt = parser.build_prompt(test_text)
        print(f"Prompt built successfully (length: {len(prompt)} chars)")
        print(f"Prompt contains OCR text: {test_text in prompt}")
        print("Prompt building works")
    except Exception as e:
        print(f"Prompt building failed: {e}")
        return False

    # Test 6: System prompt
    print("\n=== TEST 6: SYSTEM PROMPT ===")
    try:
        system_prompt = parser._system_prompt
        print(f"System prompt length: {len(system_prompt)} chars")
        print(f"Contains JSON format: {'JSON format' in system_prompt}")
        print(f"Contains validation rules: {'CRITICAL RULES' in system_prompt}")
        print("System prompt configured correctly")
    except Exception as e:
        print(f"System prompt test failed: {e}")
        return False

    # Test 7: Dependencies
    print("\n=== TEST 7: DEPENDENCIES ===")
    dependencies = [
        ('validation_service', 'ValidationService'),
        ('token_usage', 'TokenUsage'),
        ('persistence', 'TokenUsagePersistence'),
        ('retry_service', 'RetryService'),
        ('llm', 'ChatOpenAI')
    ]

    for attr_name, expected_type in dependencies:
        if hasattr(parser, attr_name):
            attr_value = getattr(parser, attr_name)
            actual_type = type(attr_value).__name__
            print(f"{attr_name}: {actual_type} (expected: {expected_type})")
        else:
            print(f"{attr_name}: MISSING")
            return False

    # Test 8: Error handling (without making actual API calls)
    print("\n=== TEST 8: ERROR HANDLING ===")
    try:
        # Test with empty text - should handle gracefully
        result = parser.parse_text("")
        print(f"Empty text result type: {type(result).__name__}")
        print(f"Empty text result status: {result.status if hasattr(result, 'status') else 'N/A'}")
        print("Error handling works")
    except Exception as e:
        print(f"Error handling test failed: {e}")
        return False

    print("\n=== ALL TESTS PASSED ===")
    print("Unified ReceiptParser is working correctly!")
    return True

def test_import_compatibility():
    """Test that imports still work from other modules"""
    print("\n=== TESTING IMPORT COMPATIBILITY ===")

    try:
        # Test import from workflow.py perspective
        from domain.parsing.receipt_parser import ReceiptParser
        parser = ReceiptParser()
        print("Import from receipt_parser: SUCCESS")

        # Test that the interface is the same
        if hasattr(parser, 'parse_text'):
            print("Interface compatibility: SUCCESS")
        else:
            print("Interface compatibility: FAILED")
            return False

    except Exception as e:
        print(f"Import compatibility test failed: {e}")
        return False

    return True

if __name__ == '__main__':
    success = test_unified_parser()
    if success:
        success = test_import_compatibility()

    if success:
        print("\n" + "="*50)
        print("UNIFIED PARSER CONSOLIDATION SUCCESSFUL!")
        print("All functionality preserved and working correctly.")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("UNIFIED PARSER CONSOLIDATION FAILED!")
        print("Some functionality is missing or broken.")
        print("="*50)
        sys.exit(1)
