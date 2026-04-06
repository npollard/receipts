"""Unit tests for ReceiptProcessor"""

import pytest
from unittest.mock import Mock, patch
from receipt_processor import ReceiptProcessor
from api_response import APIResponse


def test_receipt_processor_initialization():
    """Test ReceiptProcessor initialization"""
    with patch('receipt_processor.VisionProcessor'), \
         patch('receipt_processor.ReceiptParser'), \
         patch('receipt_processor.WorkflowOrchestrator'):
        
        processor = ReceiptProcessor()
        
        assert processor.image_processor is not None
        assert processor.ai_parser is not None
        assert processor.orchestrator is not None


def test_process_directly_success():
    """Test direct processing success"""
    expected_response = APIResponse.success({"data": "value"})
    
    with patch('receipt_processor.VisionProcessor'), \
         patch('receipt_processor.ReceiptParser'), \
         patch('receipt_processor.WorkflowOrchestrator') as mock_orchestrator:
        
        mock_orchestrator.return_value.process_image.return_value = expected_response
        
        processor = ReceiptProcessor()
        result = processor.process_directly("test.jpg")
        
        assert result == expected_response
        processor.orchestrator.process_image.assert_called_once_with("test.jpg")


def test_process_directly_failure():
    """Test direct processing failure"""
    expected_response = APIResponse.failure("Processing failed")
    
    with patch('receipt_processor.VisionProcessor'), \
         patch('receipt_processor.ReceiptParser'), \
         patch('receipt_processor.WorkflowOrchestrator') as mock_orchestrator:
        
        mock_orchestrator.return_value.process_image.return_value = expected_response
        
        processor = ReceiptProcessor()
        result = processor.process_directly("test.jpg")
        
        assert result == expected_response
        processor.orchestrator.process_image.assert_called_once_with("test.jpg")


def test_get_token_usage_summary():
    """Test getting token usage summary"""
    expected_summary = "Input tokens: 100\nOutput tokens: 50"
    
    with patch('receipt_processor.VisionProcessor'), \
         patch('receipt_processor.ReceiptParser'), \
         patch('receipt_processor.WorkflowOrchestrator') as mock_orchestrator:
        
        mock_orchestrator.return_value.get_token_usage_summary.return_value = expected_summary
        
        processor = ReceiptProcessor()
        result = processor.get_token_usage_summary()
        
        assert result == expected_summary
        processor.orchestrator.get_token_usage_summary.assert_called_once()


def test_reset_token_usage():
    """Test resetting token usage"""
    with patch('receipt_processor.VisionProcessor'), \
         patch('receipt_processor.ReceiptParser'), \
         patch('receipt_processor.WorkflowOrchestrator') as mock_orchestrator:
        
        # Mock the token_usage attribute
        mock_token_usage = Mock()
        mock_orchestrator.return_value.token_usage = mock_token_usage
        
        processor = ReceiptProcessor()
        processor.reset_token_usage()
        
        # Verify a new TokenUsage was created and assigned
        assert processor.orchestrator.token_usage != mock_token_usage
