"""Unit tests for ReceiptParser"""

import pytest
import os
from unittest.mock import Mock, patch
from domain.parsing.receipt_parser import ReceiptParser
from api_response import APIResponse


def test_receipt_parser_initialization():
    """Test ReceiptParser initialization with default settings"""
    with patch('domain.parsing.receipt_parser.ChatOpenAI'):
        parser = ReceiptParser()
        assert parser.llm is not None
        assert parser.persistence is not None


def test_receipt_parser_initialization_custom_settings():
    """Test ReceiptParser initialization with custom settings"""
    with patch('domain.parsing.receipt_parser.ChatOpenAI') as mock_llm, \
         patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        ReceiptParser(model_name="gpt-4o", temperature=0.5)
        mock_llm.assert_called_once_with(
            model="gpt-4o",
            temperature=0.5,
            api_key="test-key"
        )


def test_build_prompt():
    """Test prompt building from OCR text"""
    with patch('domain.parsing.receipt_parser.ChatOpenAI'), \
         patch('domain.parsing.receipt_parser.TokenUsagePersistence'):

        parser = ReceiptParser()
        ocr_text = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"

        prompt = parser.build_prompt(ocr_text)

        assert "MILK 4.50" in prompt
        assert "BREAD 3.00" in prompt
        assert "TOTAL 7.50" in prompt
        assert "JSON" in prompt
        assert "receipt" in prompt.lower()


def test_parse_text_success():
    """Test successful parsing"""
    ocr_text = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"
    expected_response = Mock()
    expected_response.content = '{"date": "2026-03-30", "total": 7.50, "items": [{"description": "Milk", "price": 4.50}, {"description": "Bread", "price": 3.00}]}'
    expected_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

    with patch('domain.parsing.receipt_parser.ChatOpenAI') as mock_llm, \
         patch('domain.parsing.receipt_parser.validate_response_content', return_value=APIResponse.success({"date": "2026-03-30", "total": 7.50, "items": [{"description": "Milk", "price": 4.50}, {"description": "Bread", "price": 3.00}]})), \
         patch('domain.parsing.receipt_parser.validate_with_pydantic', return_value=APIResponse.success({"date": "2026-03-30", "total": 7.50, "items": [{"description": "Milk", "price": 4.50}, {"description": "Bread", "price": 3.00}], "_token_usage": {"input_tokens": 100, "output_tokens": 50}})), \
         patch('domain.parsing.receipt_parser.TokenUsagePersistence'):

        mock_llm.return_value.invoke.return_value = expected_response

        parser = ReceiptParser()
        result = parser.parse_text(ocr_text)

        assert result.status == "success"
        # result.data is now a Pydantic model, not a dictionary
        assert hasattr(result.data, 'date')
        assert hasattr(result.data, 'total')
        assert hasattr(result.data, 'items')


def test_parse_text_pydantic_validation_failure():
    """Test parsing when Pydantic validation fails"""
    ocr_text = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"
    expected_response = Mock()
    expected_response.content = '{"date": "invalid-date", "total": "not-a-number", "items": []}'
    expected_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

    with patch('domain.parsing.receipt_parser.ChatOpenAI') as mock_llm, \
         patch('domain.parsing.receipt_parser.validate_response_content', return_value=APIResponse.success({"date": "invalid-date", "total": "not-a-number", "items": []})), \
         patch('domain.parsing.receipt_parser.validate_with_pydantic', return_value=APIResponse.failure("Validation error")), \
         patch('domain.parsing.receipt_parser.TokenUsagePersistence'):

        mock_llm.return_value.invoke.return_value = expected_response

        parser = ReceiptParser()
        result = parser.parse_text(ocr_text)

        assert result.status == "failed"
        assert "Data validation failed" in result.error


def test_parse_text_api_exception():
    """Test parsing when API call fails"""
    ocr_text = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"

    with patch('domain.parsing.receipt_parser.ChatOpenAI') as mock_llm, \
         patch('domain.parsing.receipt_parser.TokenUsagePersistence'):

        mock_llm.return_value.invoke.side_effect = Exception("API Error")

        parser = ReceiptParser()
        result = parser.parse_text(ocr_text)

        assert result.status == "failed"
        assert "Parsing error" in result.error


def test_parse_with_retry_success_on_first_attempt():
    """Test parsing with retry service - success on first attempt"""
    ocr_text = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"
    expected_receipt = {"date": "2026-03-30", "total": 7.50, "items": [], "_token_usage": {"input_tokens": 100, "output_tokens": 50}}

    with patch('domain.parsing.receipt_parser.ChatOpenAI'), \
         patch('domain.parsing.receipt_parser.TokenUsagePersistence'):

        parser = ReceiptParser()
        token_usage = Mock()


def test_get_token_usage_safely():
    """Test safe token usage extraction"""
    with patch('domain.parsing.receipt_parser.ChatOpenAI'), \
         patch('domain.parsing.receipt_parser.TokenUsagePersistence'):

        parser = ReceiptParser()

        # Test with valid data
        data_with_tokens = {"_token_usage": {"input_tokens": 100, "output_tokens": 50}}
        result = parser._get_token_usage_safely(data_with_tokens, "input_tokens")
        assert result == 100

        # Test with missing data
        result_none = parser._get_token_usage_safely(None, "input_tokens")
        assert result_none == 0

        # Test with missing token usage
        data_no_tokens = {"items": []}
        result_no_tokens = parser._get_token_usage_safely(data_no_tokens, "input_tokens")
        assert result_no_tokens == 0
