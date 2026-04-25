"""Core receipt parsing functionality"""

from typing import Callable, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from contracts.interfaces import LanguageModelInterface, ReceiptParsingInterface
from domain.models.receipt import Receipt
from api_response import APIResponse
from tracking import TokenUsage, extract_token_usage
from domain.validation.validation_service import ValidationService
from domain.validation.validation_utils import validate_response_content, validate_with_pydantic
from domain.parsing.parsing_result import ParsingResult
from domain.parsing.retry_strategies import ParserRetryStrategies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.retry_service import RetryService
from core.logging import get_parser_logger
from contracts.interfaces import ImageProcessingInterface

logger = get_parser_logger(__name__)


def create_default_llm(model_name: str, temperature: float) -> LanguageModelInterface:
    from infrastructure.llm import create_openai_chat_llm

    return create_openai_chat_llm(model_name=model_name, temperature=temperature)


class ReceiptParser(ReceiptParsingInterface):
    """Unified receipt parser with LLM-based parsing and retry logic"""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0,
                 ocr_service: ImageProcessingInterface = None,
                 retry_service: 'RetryService' = None,
                 llm: Optional[LanguageModelInterface] = None,
                 llm_factory: Optional[Callable[..., LanguageModelInterface]] = None):
        factory = llm_factory or create_default_llm
        self.llm = llm or factory(model_name=model_name, temperature=temperature)
        self.validation_service = ValidationService()
        self.retry_service = retry_service  # Injected, for retry orchestration
        self.token_usage = TokenUsage()
        self.ocr_service = ocr_service  # Injected, for OCR fallback retry
        self.current_retries = []  # Track retry strategies used

        self._system_prompt = """
You are a receipt parser that returns ONLY valid JSON.

CRITICAL RULES:
- Return ONLY: JSON object
- No explanations, no markdown, no code blocks
- All prices must be numbers
- If you cannot extract data, return: {"date": null, "total": null, "items": []}

Expected JSON format:
{
    "date": "YYYY-MM-DD",
    "total": 123.45,
    "items": [
        {
            "description": "Item description",
            "price": 12.34
        }
    ]
}
"""
        logger.debug(f"Initialized ReceiptParser with model: {model_name}")

    def _classify_validation_error(self, error_message: str) -> Dict[str, Any]:
        return self._retry_strategies().classify_validation_error(error_message)

    def _llm_self_correction_retry(self, original_text: str, previous_result: dict, error_info: dict) -> ParsingResult:
        return self._retry_strategies().llm_self_correction_retry(original_text, previous_result, error_info)

    def _rag_retry_with_focused_context(self, original_text: str, error_info: dict) -> ParsingResult:
        return self._retry_strategies().rag_retry_with_focused_context(original_text, error_info)

    def _ocr_fallback_retry(self, image_path: str, error_info: dict) -> ParsingResult:
        return self._retry_strategies().ocr_fallback_retry(image_path, error_info)

    def _retry_strategies(self) -> ParserRetryStrategies:
        return ParserRetryStrategies(
            llm=self.llm,
            system_prompt=self._system_prompt,
            token_usage=self.token_usage,
            current_retries=self.current_retries,
            ocr_service=self.ocr_service,
        )

    def parse_with_validation_driven_retry(self, ocr_text: str, image_path: str = None) -> ParsingResult:
        """Main parsing method with validation-driven retry strategy

        Always returns ParsingResult with the best parsed data found, even if validation failed.
        Tracks and accumulates token usage across all attempts.
        """
        max_retries = 2  # Max 2 total retries per receipt
        retry_count = 0

        # Track the best attempt (even if invalid) and accumulated token usage
        best_attempt = ParsingResult()
        accumulated_token_usage = TokenUsage()

        # Reset retry tracking for this parsing session
        self.current_retries = []

        # First attempt: normal parsing
        logger.debug("Starting validation-driven retry parsing")

        try:
            # First attempt
            first_result = self.parse_text(ocr_text)

            # Accumulate token usage from first attempt
            accumulated_token_usage.add_usage(
                first_result.token_usage.input_tokens,
                first_result.token_usage.output_tokens
            )

            # Track best attempt (prioritize valid results, but keep any parsed data)
            if first_result.parsed is not None:
                best_attempt = first_result

            # If first attempt is successful, return immediately
            if first_result.valid:
                logger.debug("Parsing successful on first attempt")
                first_result.token_usage = accumulated_token_usage
                return first_result

            # First attempt failed validation, classify error for retry
            error_info = self._classify_validation_error(first_result.error or "Validation failed")
            logger.debug(f"Validation error classified: {error_info}")

            retry_count = 1

            # Retry Strategy 1: LLM Self-Correction (for small/medium errors)
            if retry_count <= max_retries and error_info['severity'] in ['small', 'medium']:
                logger.debug(f"Retry {retry_count}/{max_retries}: LLM self-correction")
                if first_result.parsed:
                    retry_result = self._llm_self_correction_retry(
                        ocr_text,
                        first_result.parsed.model_dump(),
                        error_info
                    )
                else:
                    # Fallback to regular retry if no previous result
                    retry_result = self.parse_with_retry(ocr_text, max_retries=1)
                    # Convert APIResponse to ParsingResult if needed
                    if isinstance(retry_result, APIResponse):
                        retry_result = ParsingResult(
                            parsed=retry_result.data if retry_result.success else None,
                            valid=retry_result.success,
                            error=retry_result.error if not retry_result.success else None,
                            token_usage=self.token_usage
                        )

                # Accumulate token usage from retry
                accumulated_token_usage.add_usage(
                    retry_result.token_usage.input_tokens,
                    retry_result.token_usage.output_tokens
                )

                # Update best attempt if this retry has parsed data
                if retry_result.parsed is not None:
                    best_attempt = retry_result

                if retry_result.valid:
                    logger.debug(f"LLM self-correction successful on retry {retry_count}")
                    retry_result.token_usage = accumulated_token_usage
                    return retry_result
                else:
                    logger.warning(f"LLM self-correction failed on retry {retry_count}")
                    retry_count += 1

            # Retry Strategy 2: RAG with Focused Context (for medium/large errors)
            if retry_count <= max_retries and error_info['severity'] in ['medium', 'large']:
                logger.debug(f"Retry {retry_count}/{max_retries}: RAG with focused context")
                retry_result = self._rag_retry_with_focused_context(ocr_text, error_info)

                # Accumulate token usage from retry
                accumulated_token_usage.add_usage(
                    retry_result.token_usage.input_tokens,
                    retry_result.token_usage.output_tokens
                )

                # Update best attempt if this retry has parsed data
                if retry_result.parsed is not None:
                    best_attempt = retry_result

                if retry_result.valid:
                    logger.debug(f"RAG retry successful on retry {retry_count}")
                    retry_result.token_usage = accumulated_token_usage
                    return retry_result
                else:
                    logger.warning(f"RAG retry failed on retry {retry_count}")
                    retry_count += 1

            # Retry Strategy 3: OCR Fallback (if still failing OR low OCR quality)
            if retry_count <= max_retries and image_path and self.ocr_service:
                # Check OCR quality
                ocr_quality = self.ocr_service.score_ocr_quality(ocr_text)
                low_quality = ocr_quality < 0.25

                if low_quality or error_info['severity'] in ['medium', 'large']:
                    logger.debug(f"Retry {retry_count}/{max_retries}: OCR fallback (quality: {ocr_quality:.3f})")
                    retry_result = self._ocr_fallback_retry(image_path, error_info)

                    # Accumulate token usage from retry
                    accumulated_token_usage.add_usage(
                        retry_result.token_usage.input_tokens,
                        retry_result.token_usage.output_tokens
                    )

                    # Update best attempt if this retry has parsed data
                    if retry_result.parsed is not None:
                        best_attempt = retry_result

                    if retry_result.valid:
                        logger.debug(f"OCR fallback successful on retry {retry_count}")
                        retry_result.token_usage = accumulated_token_usage
                        return retry_result
                    else:
                        logger.warning(f"OCR fallback failed on retry {retry_count}")
                        retry_count += 1

            # All retries exhausted - return best attempt with accumulated token usage
            logger.error(f"All {max_retries} retry attempts failed, returning best attempt")
            if best_attempt.parsed is not None:
                # Return best attempt with accumulated token usage and error info
                best_attempt.token_usage = accumulated_token_usage
                if not best_attempt.error:
                    best_attempt.error = f"Validation failed after {max_retries} retries"
                return best_attempt
            else:
                # No parsed data at all, return failure
                return ParsingResult(
                    parsed=None,
                    valid=False,
                    error=f"Validation-driven retry failed after {max_retries} attempts. Last error: {error_info['error_message']}",
                    token_usage=accumulated_token_usage
                )

        except Exception as e:
            logger.error(f"Unexpected error in validation-driven retry: {str(e)}")
            # Even on exception, return best attempt if we have one
            if best_attempt.parsed is not None:
                best_attempt.token_usage = accumulated_token_usage
                if not best_attempt.error:
                    best_attempt.error = f"Retry strategy failed: {str(e)}"
                return best_attempt
            return ParsingResult(
                parsed=None,
                valid=False,
                error=f"Retry strategy failed: {str(e)}",
                token_usage=accumulated_token_usage
            )

    def get_current_retries(self) -> list:
        """Get list of retry strategies used in current parsing session"""
        return self.current_retries.copy()

    def build_prompt(self, ocr_text: str) -> str:
        """Build parsing prompt from OCR text"""
        return f"""
Extract receipt data from this OCR text and return ONLY valid JSON:

OCR TEXT:
{ocr_text}

Remember: Return ONLY: JSON object, nothing else.
"""

    def parse_text(self, text: str) -> ParsingResult:
        """Main parsing method for OCR text into structured receipt data

        Returns ParsingResult that preserves parsed data even on validation failure
        """
        result = ParsingResult()

        try:
            # Validate input
            if not text or not text.strip():
                result.error = "Empty text provided for parsing"
                return result

            logger.debug(f"Parsing OCR text of length: {len(text)}")
            logger.debug(f"OCR text content: {repr(text[:200])}...")

            prompt = self.build_prompt(text)
            messages = [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=prompt)
            ]

            # Invoke LLM with specific error handling
            try:
                response = self.llm.invoke(messages)
            except Exception as e:
                result.error = f"Failed to invoke AI model: {str(e)}"
                return result

            # Extract token usage from response
            try:
                input_tokens, output_tokens, total_tokens = extract_token_usage(response)
                logger.debug(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")
                result.token_usage.add_usage(input_tokens, output_tokens)
                self.token_usage.add_usage(input_tokens, output_tokens)
            except Exception as e:
                result.error = f"Failed to extract token usage: {str(e)}"
                return result

            # Validate response content
            try:
                parsed_response = validate_response_content(response)
            except Exception as e:
                result.error = f"Failed to validate response content: {str(e)}"
                return result

            if parsed_response.status == "failed":
                # Preserve parsed data even if validation failed
                logger.debug(f"Response content validation failed. parsed_response.data: {parsed_response.data}")
                if parsed_response.data:
                    result.parsed = parsed_response.data
                    logger.info(f"Preserved parsed data on content validation failure: {parsed_response.data}")
                result.error = f"Response content validation failed: {parsed_response.error}"
                return result

            # Validate with Pydantic model
            try:
                validated_response = validate_with_pydantic(
                    parsed_response.data,
                    input_tokens,
                    output_tokens
                )
            except ValidationError as e:
                # Preserve parsed data even if Pydantic validation failed
                result.parsed = parsed_response.data
                result.error = "Validation failed"
                return result
            except Exception as e:
                # Preserve parsed data even if validation failed
                result.parsed = parsed_response.data
                result.error = "Validation failed"
                return result

            if validated_response.status == "failed":
                # Preserve parsed data even if validation failed
                logger.debug(f"Validation failed. parsed_response.data: {parsed_response.data}")
                result.parsed = parsed_response.data
                logger.info(f"Preserved parsed data on validation failure: {parsed_response.data}")
                result.error = "Validation failed"
                return result

            # Success case
            logger.info("Parsing successful on attempt 1")
            # Convert dict to Pydantic model, extracting token usage first
            data = validated_response.data.copy()
            data.pop("_token_usage", None)  # Remove token usage before model creation
            receipt = Receipt(**data)
            result.parsed = receipt
            result.valid = True
            result.error = None
            return result

        except Exception as e:
            result.error = f"Unexpected error during receipt parsing: {str(e)}"
            return result

    def parse_with_retry(self, text: str, token_usage=None, max_retries: int = 3) -> APIResponse:
        """Parse with retry logic and error fixing"""
        if token_usage is None:
            token_usage = self.token_usage

        def parse_attempt(*args, **kwargs):
            return self.parse_text(text)

        def fix_prompt_generator(original_text, error, attempt):
            return self._create_fix_prompt(original_text, str(error), attempt)

        # Use retry service with error fixing if injected, otherwise direct execution
        if self.retry_service:
            return self.retry_service.execute_with_retry_and_fix(
                parse_attempt,
                fix_prompt_generator,
                text,
                error_types=(Exception,)
            )
        else:
            return self.parse_text(text)

    def parse_with_usage_tracking(self, text: str, token_usage=None) -> APIResponse:
        """Parse text with token usage tracking"""
        if token_usage is None:
            token_usage = self.token_usage

        result = self.parse_with_retry(text, token_usage)
        # Note: Token usage persistence moved to application layer
        return result

    def _create_fix_prompt(self, original_text: str, error_message: str, attempt_count: int) -> str:
        """Create targeted prompt to fix specific parsing errors"""
        return f"""
The previous parsing attempt failed with this error:
{error_message}

Please fix this error and re-parse the receipt OCR text:

OCR TEXT:
{original_text}

Return ONLY valid JSON that addresses the specific error mentioned above.
"""

    def _get_token_usage_safely(self, data: dict, key: str) -> int:
        """Safely extract token usage from response data"""
        try:
            token_usage = data.get('_token_usage', {})
            return int(token_usage.get(key, 0))
        except (AttributeError, ValueError, TypeError):
            return 0

    @tool
    def parse_receipt_text(self, ocr_text: str) -> Dict[str, Any]:
        """LangChain tool for receipt parsing"""
        result = self.parse_with_usage_tracking(ocr_text)

        if result.success:
            return result.data
        else:
            return {"error": result.error, "status": "failed"}

    def get_token_usage_summary(self) -> str:
        """Get current token usage summary"""
        return self.token_usage.get_summary()

    def get_token_usage(self) -> TokenUsage:
        """Get token usage from parsing operations"""
        return self.token_usage

    def reset_token_usage(self):
        """Reset token usage tracking"""
        self.token_usage = TokenUsage()
