"""Unit tests for ReceiptParser"""

from unittest.mock import Mock, patch
from domain.parsing.receipt_parser import ReceiptParser
from api_response import APIResponse


def _fake_llm(response=None, side_effect=None):
    llm = Mock()
    if side_effect is not None:
        llm.invoke.side_effect = side_effect
    else:
        llm.invoke.return_value = response or Mock(
            content='{"date": null, "total": null, "items": []}',
            usage_metadata={"input_tokens": 0, "output_tokens": 0},
        )
    return llm


def test_receipt_parser_initialization():
    """Test ReceiptParser initialization with default settings"""
    parser = ReceiptParser(llm=_fake_llm())
    assert parser.llm is not None
    assert parser.ocr_service is None  # No OCR service injected by default


def test_receipt_parser_initialization_custom_settings():
    """Test ReceiptParser initialization with custom settings"""
    llm = _fake_llm()
    factory = Mock(return_value=llm)

    parser = ReceiptParser(model_name="gpt-4o", temperature=0.5, llm_factory=factory)

    assert parser.llm is llm
    factory.assert_called_once_with(model_name="gpt-4o", temperature=0.5)


def test_receipt_parser_uses_default_llm_factory_when_not_injected():
    llm = _fake_llm()

    with patch("domain.parsing.receipt_parser.create_default_llm", return_value=llm) as factory:
        parser = ReceiptParser(model_name="gpt-4o", temperature=0.5)

    assert parser.llm is llm
    factory.assert_called_once_with(model_name="gpt-4o", temperature=0.5)


def test_build_prompt():
    """Test prompt building from OCR text"""
    parser = ReceiptParser(llm=_fake_llm())
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

    with patch('domain.parsing.receipt_parser.validate_response_content', return_value=APIResponse.success({"date": "2026-03-30", "total": 7.50, "items": [{"description": "Milk", "price": 4.50}, {"description": "Bread", "price": 3.00}]})), \
         patch('domain.parsing.receipt_parser.validate_with_pydantic', return_value=APIResponse.success({"date": "2026-03-30", "total": 7.50, "items": [{"description": "Milk", "price": 4.50}, {"description": "Bread", "price": 3.00}], "_token_usage": {"input_tokens": 100, "output_tokens": 50}})):

        parser = ReceiptParser(llm=_fake_llm(expected_response))
        result = parser.parse_text(ocr_text)

        assert result.valid is True
        # result.parsed is now a Pydantic model, not a dictionary
        assert hasattr(result.parsed, 'date')
        assert hasattr(result.parsed, 'total')
        assert hasattr(result.parsed, 'items')


def test_parse_text_pydantic_validation_failure():
    """Test parsing when Pydantic validation fails"""
    ocr_text = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"
    expected_response = Mock()
    expected_response.content = '{"date": "invalid-date", "total": "not-a-number", "items": []}'
    expected_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

    with patch('domain.parsing.receipt_parser.validate_response_content', return_value=APIResponse.success({"date": "invalid-date", "total": "not-a-number", "items": []})), \
         patch('domain.parsing.receipt_parser.validate_with_pydantic', return_value=APIResponse.failure("Validation error")):

        parser = ReceiptParser(llm=_fake_llm(expected_response))
        result = parser.parse_text(ocr_text)

        assert result.valid is False
        assert "Validation failed" in result.error


def test_parse_text_api_exception():
    """Test parsing when API call fails"""
    ocr_text = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"

    parser = ReceiptParser(llm=_fake_llm(side_effect=Exception("API Error")))
    result = parser.parse_text(ocr_text)

    assert result.valid is False
    assert result.error is not None


def test_parse_with_retry_success_on_first_attempt():
    """Test parsing with retry service - success on first attempt"""
    ocr_text = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"
    expected_receipt = {"date": "2026-03-30", "total": 7.50, "items": [], "_token_usage": {"input_tokens": 100, "output_tokens": 50}}

    parser = ReceiptParser(llm=_fake_llm())
    token_usage = Mock()


def test_get_token_usage_safely():
    """Test safe token usage extraction"""
    parser = ReceiptParser(llm=_fake_llm())

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
