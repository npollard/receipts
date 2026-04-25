from unittest.mock import Mock, patch

from api_response import APIResponse
from domain.parsing.retry_strategies import ParserRetryStrategies
from tracking import TokenUsage


def _strategies(llm=None, ocr_service=None):
    retries = []
    usage = TokenUsage()
    return ParserRetryStrategies(
        llm=llm or Mock(),
        system_prompt="system",
        token_usage=usage,
        current_retries=retries,
        ocr_service=ocr_service,
    ), retries, usage


def test_classifies_total_mismatch_by_size():
    strategies, _, _ = _strategies()

    assert strategies.classify_validation_error(
        "Total 7.50 does not match sum of items 7.00"
    )["severity"] == "small"
    assert strategies.classify_validation_error(
        "Total 12.00 does not match sum of items 8.00"
    )["severity"] == "medium"
    assert strategies.classify_validation_error(
        "Total 30.00 does not match sum of items 8.00"
    )["severity"] == "large"


def test_classifies_non_total_validation_errors():
    strategies, _, _ = _strategies()

    assert strategies.classify_validation_error("Missing date")["severity"] == "medium"
    assert strategies.classify_validation_error("Invalid price")["severity"] == "small"
    assert strategies.classify_validation_error("Could not parse")["severity"] == "unknown"


def test_llm_self_correction_invokes_model_and_tracks_retry():
    response = Mock(usage_metadata={"input_tokens": 11, "output_tokens": 7})
    llm = Mock()
    llm.invoke.return_value = response
    strategies, retries, usage = _strategies(llm=llm)

    with patch(
        "domain.parsing.retry_strategies.validate_response_content",
        return_value=APIResponse.success({"date": "2026-03-30", "total": 7.5, "items": []}),
    ), patch(
        "domain.parsing.retry_strategies.validate_with_pydantic",
        return_value=APIResponse.success({"date": "2026-03-30", "total": 7.5, "items": []}),
    ):
        result = strategies.llm_self_correction_retry(
            "OCR text",
            {"total": 7.0},
            {"severity": "small", "error_message": "Invalid total"},
        )

    assert result.valid is True
    assert retries == ["LLM_SELF_CORRECTION"]
    assert usage.input_tokens == 11
    assert usage.output_tokens == 7


def test_rag_retry_uses_focused_context_and_tracks_retry():
    response = Mock(usage_metadata={"input_tokens": 5, "output_tokens": 3})
    llm = Mock()
    llm.invoke.return_value = response
    strategies, retries, _ = _strategies(llm=llm)

    with patch(
        "domain.parsing.retry_strategies.validate_response_content",
        return_value=APIResponse.failure("bad json", data={"raw": True}),
    ):
        result = strategies.rag_retry_with_focused_context(
            "HEADER\nAPPLES 3.00\nTOTAL 3.00",
            {"severity": "large", "error_message": "Total mismatch"},
        )

    assert result.valid is False
    assert result.parsed == {"raw": True}
    assert result.error == "RAG retry failed: bad json"
    assert retries == ["RAG_FOCUSED_CONTEXT"]
    sent_prompt = llm.invoke.call_args[0][0][1].content
    assert "TOTAL 3.00" in sent_prompt


def test_ocr_fallback_requires_ocr_service():
    strategies, retries, _ = _strategies()

    result = strategies.ocr_fallback_retry("receipt.jpg", {"severity": "large"})

    assert result.valid is False
    assert result.error == "OCR service not available for fallback"
    assert retries == []


def test_ocr_fallback_invokes_vision_ocr_then_model():
    response = Mock(usage_metadata={"input_tokens": 4, "output_tokens": 2})
    llm = Mock()
    llm.invoke.return_value = response
    ocr = Mock()
    ocr.extract_text.return_value = "VISION TOTAL 9.00"
    strategies, retries, _ = _strategies(llm=llm, ocr_service=ocr)

    with patch(
        "domain.parsing.retry_strategies.validate_response_content",
        return_value=APIResponse.success({"date": "2026-03-30", "total": 9.0, "items": []}),
    ), patch(
        "domain.parsing.retry_strategies.validate_with_pydantic",
        return_value=APIResponse.success({"date": "2026-03-30", "total": 9.0, "items": []}),
    ):
        result = strategies.ocr_fallback_retry("receipt.jpg", {"severity": "large"})

    assert result.valid is True
    assert retries == ["OCR_FALLBACK"]
    ocr.extract_text.assert_called_once_with("receipt.jpg", use_vision_fallback=True)
