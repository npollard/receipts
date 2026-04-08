"""Core receipt parsing functionality"""

import logging
from typing import Dict, Any
import json
import os
from decimal import Decimal
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from contracts.interfaces import ReceiptParsingInterface
from models.receipt import Receipt
from api_response import APIResponse
from token_usage_persistence import TokenUsagePersistence
from tracking import TokenUsage, extract_token_usage
from domain.validation.validation_service import ValidationService
from domain.validation.validation_utils import validate_response_content, validate_with_pydantic, handle_validation_error
from services.retry_service import RetryService, RetryStrategy

logger = logging.getLogger(__name__)


class ReceiptParser(ReceiptParsingInterface):
    """Unified receipt parser with LLM-based parsing and retry logic"""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.validation_service = ValidationService()
        self.retry_service = RetryService()
        self.token_usage = TokenUsage()
        self.persistence = TokenUsagePersistence()

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
        logger.info(f"Initialized ReceiptParser with model: {model_name}")

    def build_prompt(self, ocr_text: str) -> str:
        """Build parsing prompt from OCR text"""
        return f"""
Extract receipt data from this OCR text and return ONLY valid JSON:

OCR TEXT:
{ocr_text}

Remember: Return ONLY: JSON object, nothing else.
"""

    def parse_text(self, text: str) -> APIResponse:
        """Main parsing method for OCR text into structured receipt data"""
        logger.debug(f"Parsing OCR text of length: {len(text)}")
        logger.debug(f"OCR text content: {repr(text[:200])}...")  # Log raw text for debugging

        prompt = self.build_prompt(text)
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            response = self.llm.invoke(messages)

            # Extract token usage from response
            input_tokens, output_tokens, total_tokens = extract_token_usage(response)
            logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")

            # Validate response content
            parsed_response = validate_response_content(response)

            if parsed_response.status == "failed":
                logger.warning(f"Response content validation failed: {parsed_response.error}")
                return parsed_response

            # Validate with Pydantic model
            validated_response = validate_with_pydantic(
                parsed_response.data,
                input_tokens,
                output_tokens
            )

            if validated_response.status == "failed":
                logger.warning(f"Pydantic validation failed: {validated_response.error}")
                return validated_response

            logger.info("Parsing successful on attempt 1")
            return validated_response

        except ValidationError as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(error_msg)
            return APIResponse.failure(error_msg)

        except Exception as e:
            error_msg = f"Parsing error: {str(e)}"
            logger.error(error_msg)
            return APIResponse.failure(error_msg)

    def parse_with_retry(self, text: str, token_usage=None, max_retries: int = 3) -> APIResponse:
        """Parse with retry logic and error fixing"""
        if token_usage is None:
            token_usage = self.token_usage

        def parse_attempt(*args, **kwargs):
            return self.parse_text(text)

        def fix_prompt_generator(original_text, error, attempt):
            return self._create_fix_prompt(original_text, str(error), attempt)

        # Use retry service with error fixing
        return self.retry_service.execute_with_retry_and_fix(
            parse_attempt,
            fix_prompt_generator,
            text,
            error_types=(Exception,)
        )

    def parse_with_usage_tracking(self, text: str, token_usage=None) -> APIResponse:
        """Parse text with token usage tracking"""
        if token_usage is None:
            token_usage = self.token_usage

        result = self.parse_with_retry(text, token_usage)

        # Save token usage if successful
        if result.success and token_usage.get_total_tokens() > 0:
            session_id = f"session_{token_usage.get_total_tokens()}"
            self.persistence.save_usage(token_usage, session_id)
            logger.info(f"Saved token usage to persistent storage: {session_id}")

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

    def reset_token_usage(self):
        """Reset token usage tracking"""
        self.token_usage = TokenUsage()
