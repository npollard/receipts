"""Unit tests for validation utilities"""

import pytest
from domain.validation.validation_utils import validate_response_content, validate_with_pydantic, handle_validation_error
from api_response import APIResponse
from models.receipt import Receipt, ReceiptItem
from pydantic import ValidationError
from unittest.mock import Mock


def test_validate_response_content_success():
    """Test successful response content validation"""
    mock_response = Mock()
    mock_response.content = '{"date": "2026-03-30", "total": 7.50, "items": []}'

    result = validate_response_content(mock_response)

    assert result.status == "success"
    assert result.data == {"date": "2026-03-30", "total": 7.50, "items": []}


def test_validate_response_content_invalid_json():
    """Test response content validation with invalid JSON"""
    mock_response = Mock()
    mock_response.content = 'not valid json'

    result = validate_response_content(mock_response)

    assert result.status == "failed"
    assert "Failed to parse JSON" in result.error


def test_validate_response_content_empty_response():
    """Test response content validation with empty response"""
    mock_response = Mock()
    mock_response.content = ''

    result = validate_response_content(mock_response)

    assert result.status == "failed"
    assert "Failed to parse JSON" in result.error


def test_validate_response_content_json_exception():
    """Test response content validation with JSON parsing exception"""
    mock_response = Mock()
    mock_response.content = '{"incomplete": json'

    result = validate_response_content(mock_response)

    assert result.status == "failed"
    assert "Failed to parse JSON" in result.error


def test_validate_response_content_non_dict():
    """Test response content validation with non-dict JSON"""
    mock_response = Mock()
    mock_response.content = '["not", "a", "dict"]'

    result = validate_response_content(mock_response)

    assert result.status == "failed"
    assert "Expected JSON object" in result.error


def test_validate_with_pydantic_success():
    """Test successful Pydantic validation"""
    parsed_data = {
        "date": "2026-03-30",
        "total": 4.50,  # Must match sum of items
        "items": [
            {"description": "Milk", "price": 4.50}
        ]
    }

    result = validate_with_pydantic(parsed_data, 100, 50)

    assert result.status == "success"
    assert "date" in result.data
    assert "total" in result.data
    assert "items" in result.data
    assert "_token_usage" in result.data
    assert result.data["_token_usage"]["input_tokens"] == 100
    assert result.data["_token_usage"]["output_tokens"] == 50


def test_validate_with_pydantic_validation_error():
    """Test Pydantic validation with validation error"""
    invalid_data = {
        "date": "invalid-date",
        "total": "not-a-number",
        "items": []
    }

    result = validate_with_pydantic(invalid_data, 100, 50)

    assert result.status == "failed"
    assert "Validation failed" in result.error


def test_validate_with_pydantic_missing_required_fields():
    """Test Pydantic validation with missing optional fields"""
    # Receipt model has optional date and total, and default empty items list
    incomplete_data = {
        "date": "2026-03-30"
        # Missing 'total' and 'items' - these are optional with defaults
    }

    result = validate_with_pydantic(incomplete_data, 100, 50)

    # This should succeed because items defaults to [] and total defaults to None
    assert result.status == "success"
    assert result.data["date"] == "2026-03-30"
    assert result.data["items"] == []
    assert result.data["total"] is None


def test_validate_with_pydantic_negative_total():
    """Test Pydantic validation with negative total"""
    invalid_data = {
        "date": "2026-03-30",
        "total": -10.00,
        "items": []
    }

    result = validate_with_pydantic(invalid_data, 100, 50)

    assert result.status == "failed"
    assert "Validation failed" in result.error


def test_handle_validation_error_with_validation_error():
    """Test handling ValidationError exception"""
    try:
        # Trigger a ValidationError
        Receipt(date="invalid", total=-1, items=[])
    except ValidationError as e:
        result = handle_validation_error(e, {"date": "invalid", "total": -1, "items": []}, 100, 50)

        assert result.status == "failed"
        assert "Validation failed" in result.error


def test_handle_validation_error_with_generic_exception():
    """Test handling generic exception - this should not happen with current implementation"""
    # The current implementation only handles ValidationError, so this test
    # verifies that we can't pass a generic exception
    exception = Exception("Some unexpected error")

    # This would fail at runtime because the function expects ValidationError
    # which has an .errors() method that regular Exception doesn't have
    try:
        handle_validation_error(exception, {"data": "value"}, 100, 50)
        # If we get here, the function accepted a generic exception (unexpected)
        assert False, "Expected AttributeError for non-ValidationError exception"
    except AttributeError as e:
        # Expected behavior - function tries to call .errors() on generic Exception
        assert "no attribute 'errors'" in str(e)


def test_validate_response_content_with_whitespace():
    """Test response content validation with whitespace"""
    mock_response = Mock()
    mock_response.content = '   {"date": "2026-03-30", "total": 7.50, "items": []}   '

    result = validate_response_content(mock_response)

    assert result.status == "success"
    assert result.data == {"date": "2026-03-30", "total": 7.50, "items": []}


def test_validate_response_content_with_newlines():
    """Test response content validation with newlines"""
    mock_response = Mock()
    mock_response.content = '{"date": "2026-03-30",\n"total": 7.50,\n"items": []}'

    result = validate_response_content(mock_response)

    assert result.status == "success"
    assert result.data == {"date": "2026-03-30", "total": 7.50, "items": []}
