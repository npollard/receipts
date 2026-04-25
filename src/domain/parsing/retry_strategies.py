import re
from decimal import Decimal
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from domain.parsing.parsing_result import ParsingResult
from domain.validation.validation_utils import validate_response_content, validate_with_pydantic
from prompts.retry_prompts import get_llm_fix_prompt, get_rag_prompt, get_vision_reparse_prompt
from tracking import TokenUsage, extract_token_usage


class ParserRetryStrategies:
    """Validation-driven parser retry strategies."""

    def __init__(
        self,
        llm: Any,
        system_prompt: str,
        token_usage: TokenUsage,
        current_retries: List[str],
        ocr_service: Optional[Any] = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.token_usage = token_usage
        self.current_retries = current_retries
        self.ocr_service = ocr_service

    def classify_validation_error(self, error_message: str) -> Dict[str, Any]:
        severity = "unknown"
        mismatch_amount = None

        total_mismatch_pattern = r"Total ([\d.,]+) does not match sum of items ([\d.,]+)"
        match = re.search(total_mismatch_pattern, error_message)

        if match:
            try:
                total = Decimal(match.group(1).replace(",", "."))
                items_sum = Decimal(match.group(2).replace(",", "."))
                mismatch_amount = abs(total - items_sum)

                if mismatch_amount < Decimal("1.00"):
                    severity = "small"
                elif mismatch_amount <= Decimal("5.00"):
                    severity = "medium"
                else:
                    severity = "large"
            except (ValueError, TypeError):
                pass
        elif "missing" in error_message.lower():
            severity = "medium"
        elif "invalid" in error_message.lower():
            severity = "small"

        return {
            "severity": severity,
            "mismatch_amount": float(mismatch_amount) if mismatch_amount else None,
            "error_message": error_message,
        }

    def llm_self_correction_retry(
        self,
        original_text: str,
        previous_result: dict,
        error_info: dict,
    ) -> ParsingResult:
        self.current_retries.append("LLM_SELF_CORRECTION")
        correction_prompt = get_llm_fix_prompt(
            error_info["error_message"],
            original_text,
            previous_result,
        )
        return self._invoke_and_validate(
            correction_prompt,
            "Self-correction",
            "Self-correction failed",
            "Self-correction validation failed",
        )

    def rag_retry_with_focused_context(self, original_text: str, error_info: dict) -> ParsingResult:
        self.current_retries.append("RAG_FOCUSED_CONTEXT")
        focused_context = self._extract_focused_context(original_text)
        rag_prompt = get_rag_prompt(focused_context, original_text)
        return self._invoke_and_validate(
            rag_prompt,
            "RAG",
            "RAG retry failed",
            "RAG validation failed",
        )

    def ocr_fallback_retry(self, image_path: str, error_info: dict) -> ParsingResult:
        result = ParsingResult()

        if self.ocr_service is None:
            result.error = "OCR service not available for fallback"
            return result

        self.current_retries.append("OCR_FALLBACK")

        try:
            fallback_text = self.ocr_service.extract_text(image_path, use_vision_fallback=True)
            if not fallback_text or fallback_text.strip() == "":
                result.error = "OCR fallback returned empty text"
                return result

            vision_prompt = get_vision_reparse_prompt(fallback_text)
            return self._invoke_and_validate(
                vision_prompt,
                "OCR fallback",
                "OCR fallback parsing failed",
                "OCR fallback validation failed",
            )
        except Exception as e:
            result.error = f"OCR fallback failed: {str(e)}"
            return result

    def _invoke_and_validate(
        self,
        prompt: str,
        exception_prefix: str,
        parse_failure_prefix: str,
        validation_failure_prefix: str,
    ) -> ParsingResult:
        result = ParsingResult()

        try:
            response = self.llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt),
            ])
            input_tokens, output_tokens, _ = extract_token_usage(response)
            result.token_usage.add_usage(input_tokens, output_tokens)
            self.token_usage.add_usage(input_tokens, output_tokens)

            parsed_response = validate_response_content(response)
            if parsed_response.status == "failed":
                if parsed_response.data:
                    result.parsed = parsed_response.data
                result.error = f"{parse_failure_prefix}: {parsed_response.error}"
                return result

            validated_response = validate_with_pydantic(
                parsed_response.data,
                input_tokens,
                output_tokens,
            )

            if validated_response.status == "failed":
                result.parsed = parsed_response.data
                result.error = f"{validation_failure_prefix}: {validated_response.error}"
                return result

            result.parsed = validated_response.data
            result.valid = True
            return result

        except Exception as e:
            result.error = f"{exception_prefix} failed: {str(e)}"
            return result

    def _extract_focused_context(self, original_text: str) -> str:
        lines = original_text.split("\n")
        patterns = [
            r".*TOTAL.*",
            r".*AMOUNT.*",
            r".*BALANCE.*",
            r".*\$?\d+\.\d{2}.*",
            r".*\d+\.\d{2}.*",
        ]
        focused_lines = [
            line.strip()
            for line in lines
            if any(re.search(pattern, line.strip(), re.IGNORECASE) for pattern in patterns)
        ]

        if not focused_lines:
            focused_lines = lines[:3] + lines[-3:] if len(lines) > 6 else lines

        return "\n".join(focused_lines)
