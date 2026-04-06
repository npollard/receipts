"""Unit tests for WorkflowOrchestrator"""

import pytest
from unittest.mock import Mock, patch
from workflow import WorkflowOrchestrator
from image_processing import ImageProcessor
from ai_parsing import ReceiptParser
from api_response import APIResponse
from token_tracking import TokenUsage


def test_workflow_orchestrator_initialization():
    """Test WorkflowOrchestrator initialization"""
    mock_image_processor = Mock(spec=ImageProcessor)
    mock_ai_parser = Mock(spec=ReceiptParser)

    orchestrator = WorkflowOrchestrator(mock_image_processor, mock_ai_parser)

    assert orchestrator.image_processor == mock_image_processor
    assert orchestrator.ai_parser == mock_ai_parser
    assert isinstance(orchestrator.token_usage, TokenUsage)


def test_process_image_success():
    """Test successful image processing workflow"""
    mock_image_processor = Mock(spec=ImageProcessor)
    mock_ai_parser = Mock(spec=ReceiptParser)

    expected_ocr = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"
    expected_receipt = {"date": "2026-03-30", "total": 7.50, "items": []}

    mock_image_processor.extract_text.return_value = expected_ocr
    mock_ai_parser.parse_with_usage_tracking.return_value = APIResponse.success(expected_receipt)

    orchestrator = WorkflowOrchestrator(mock_image_processor, mock_ai_parser)
    result = orchestrator.process_image("test.jpg")

    assert result.status == "success"
    assert result.data["image_path"] == "test.jpg"
    assert result.data["ocr_text"] == expected_ocr
    assert result.data["parsed_receipt"] == expected_receipt

    mock_image_processor.extract_text.assert_called_once_with("test.jpg")
    mock_ai_parser.parse_with_usage_tracking.assert_called_once_with(expected_ocr, orchestrator.token_usage)


def test_process_image_parsing_failure():
    """Test image processing workflow with parsing failure"""
    mock_image_processor = Mock(spec=ImageProcessor)
    mock_ai_parser = Mock(spec=ReceiptParser)

    expected_ocr = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"

    mock_image_processor.extract_text.return_value = expected_ocr
    mock_ai_parser.parse_with_usage_tracking.return_value = APIResponse.failure("Parsing error")

    orchestrator = WorkflowOrchestrator(mock_image_processor, mock_ai_parser)
    result = orchestrator.process_image("test.jpg")

    assert result.status == "failed"
    assert "Parsing error" in result.error

    mock_image_processor.extract_text.assert_called_once_with("test.jpg")
    mock_ai_parser.parse_with_usage_tracking.assert_called_once_with(expected_ocr, orchestrator.token_usage)


def test_process_image_extraction_exception():
    """Test image processing workflow with extraction exception"""
    mock_image_processor = Mock(spec=ImageProcessor)
    mock_ai_parser = Mock(spec=ReceiptParser)

    mock_image_processor.extract_text.side_effect = Exception("OCR failed")

    orchestrator = WorkflowOrchestrator(mock_image_processor, mock_ai_parser)
    result = orchestrator.process_image("test.jpg")

    assert result.status == "failed"
    assert "Processing error" in result.error
    assert "OCR failed" in result.error

    mock_image_processor.extract_text.assert_called_once_with("test.jpg")
    mock_ai_parser.parse_with_usage_tracking.assert_not_called()


def test_process_image_parsing_exception():
    """Test image processing workflow with parsing exception"""
    mock_image_processor = Mock(spec=ImageProcessor)
    mock_ai_parser = Mock(spec=ReceiptParser)

    expected_ocr = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"

    mock_image_processor.extract_text.return_value = expected_ocr
    mock_ai_parser.parse_with_usage_tracking.side_effect = Exception("Parsing failed")

    orchestrator = WorkflowOrchestrator(mock_image_processor, mock_ai_parser)
    result = orchestrator.process_image("test.jpg")

    assert result.status == "failed"
    assert "Processing error" in result.error
    assert "Parsing failed" in result.error

    mock_image_processor.extract_text.assert_called_once_with("test.jpg")
    mock_ai_parser.parse_with_usage_tracking.assert_called_once_with(expected_ocr, orchestrator.token_usage)


def test_get_token_usage_summary():
    """Test getting token usage summary"""
    mock_image_processor = Mock(spec=ImageProcessor)
    mock_ai_parser = Mock(spec=ReceiptParser)

    orchestrator = WorkflowOrchestrator(mock_image_processor, mock_ai_parser)
    orchestrator.token_usage.add_usage(100, 50)

    summary = orchestrator.get_token_usage_summary()

    assert "Input Tokens: 100" in summary
    assert "Output Tokens: 50" in summary
    assert "Total Tokens: 150" in summary
