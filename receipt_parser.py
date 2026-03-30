"""Core receipt parsing functionality"""

import logging
from typing import Dict, Any
import json
import os
from decimal import Decimal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from models import Receipt
from api_response import APIResponse
from token_usage_persistence import TokenUsagePersistence
from token_utils import extract_token_usage
from validation_utils import validate_response_content, validate_with_pydantic, handle_validation_error

logger = logging.getLogger(__name__)


class ReceiptParser:
    """Concrete implementation of receipt parsing using OpenAI with persistence"""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
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
        self.persistence = TokenUsagePersistence()
        logger.info(f"Initialized ReceiptParser with model: {model_name}")

    def _build_prompt(self, ocr_text: str) -> str:
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

        prompt = self._build_prompt(text)
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            response = self.llm.invoke(messages)

            # Extract token usage from response
            input_tokens, output_tokens, total_tokens = extract_token_usage(response)
            logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")

            # Validate and extract response content
            parsed_response = validate_response_content(response)
            if parsed_response.status == "failure":
                logger.warning(f"Initial JSON parsing failed: {parsed_response.error}")
                return parsed_response

            # Validate with Pydantic
            return validate_with_pydantic(parsed_response.data, input_tokens, output_tokens)

        except Exception as e:
            logger.error(f"Unexpected parsing error: {str(e)}")
            return APIResponse.failure(f"Parsing error: {str(e)}")

    def parse_with_retry(self, text: str, token_usage, max_retries: int = 3) -> APIResponse:
        """Parse with retry mechanism and agent loop for fixing errors"""
        logger.debug(f"Starting parsing with up to {max_retries} retries")

        # First attempt
        result = self.parse_text(text)

        if result.status == "success":
            logger.info("Parsing successful on attempt 1")
            # Update token usage tracker if successful
            if result.data and "_token_usage" in result.data:
                token_usage.add_usage(
                    result.data["_token_usage"]["input_tokens"],
                    result.data["_token_usage"]["output_tokens"]
                )
            return result

        # If failed, use agent loop to fix errors
        logger.warning(f"Initial parsing failed: {result.error}")
        return self._attempt_fixes(text, result, token_usage, max_retries - 1)

    def _attempt_fixes(self, original_text: str, failed_response: APIResponse, token_usage, remaining_retries: int, attempt_count: int = 1) -> APIResponse:
        """Attempt to fix parsing errors using an agent loop"""
        logger.info(f"DEBUG: _attempt_fixes called with remaining_retries={remaining_retries}, attempt_count={attempt_count}")
        logger.info(f"DEBUG: failed_response.error = {failed_response.error}")

        if remaining_retries <= 0:
            logger.error("Max retries reached, returning failure")
            return failed_response

        logger.info(f"Attempting to fix parsing errors, {remaining_retries} retries remaining (attempt {attempt_count})")

        # Check if we're getting the same error repeatedly
        if "Field 'items': Field required" in failed_response.error and attempt_count > 2:
            logger.warning("Detected repeated 'items' field error, providing fallback response")
            fallback_response = {
                "date": None,
                "total": None,
                "items": []
            }
            logger.info(f"DEBUG: Returning fallback response: {fallback_response}")
            return APIResponse.success(fallback_response)

        # Create a prompt to fix the errors
        fix_prompt = self._create_fix_prompt(original_text, failed_response, attempt_count)
        logger.info(f"DEBUG: Created fix prompt for attempt {attempt_count}")

        try:
            # Create a new LLM call for fixing
            response = self.llm.invoke([
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=fix_prompt)
            ])

            # Extract token usage from fix attempt
            input_tokens, output_tokens, total_tokens = extract_token_usage(response)
            logger.info(f"Fix attempt token usage - Input: {input_tokens}, Output: {output_tokens}")

            # Validate the fixed response
            parsed_response = validate_response_content(response)
            if parsed_response.status == "success":
                logger.info(f"Parsing successful on fix attempt")
                # Add token usage from fix attempt to tracker
                if parsed_response.data and "_token_usage" in parsed_response.data:
                    token_usage.add_usage(
                        parsed_response.data["_token_usage"]["input_tokens"],
                        parsed_response.data["_token_usage"]["output_tokens"]
                    )
                return parsed_response
            else:
                logger.warning(f"Fix attempt failed: {parsed_response.error}")
                # Recursively try to fix again
                return self._attempt_fixes(original_text, parsed_response, token_usage, remaining_retries - 1, attempt_count + 1)

        except Exception as e:
            logger.error(f"Fix attempt {attempt_count} failed: {str(e)}")
            return APIResponse.failure(f"Fix attempt {attempt_count} failed: {str(e)}")

    def _create_fix_prompt(self, original_text: str, failed_response: APIResponse, attempt_count: int = 1) -> str:
        """Create a prompt to fix parsing errors"""
        additional_instruction = ""
        if attempt_count > 2 and "Field 'items': Field required" in failed_response.error:
            additional_instruction = """
IMPORTANT: The 'items' field is REQUIRED and must be an array.
If you cannot find any items on the receipt, return: {"date": null, "total": null, "items": []}
"""

        return f"""
The previous parsing attempt failed. Fix these issues and return valid JSON:

FAILED RESPONSE:
{failed_response.get('data', '')}

ERRORS TO FIX:
{failed_response.get('error', 'Unknown error')}

ORIGINAL OCR TEXT:
{original_text}
{additional_instruction}
Return ONLY valid JSON that fixes all the issues.
"""

    def _get_token_usage_safely(self, data: dict, key: str) -> int:
        """Safely extract token usage from response data"""
        if data is None:
            return 0
        token_usage = data.get('_token_usage', {})
        if token_usage is None:
            return 0
        return token_usage.get(key, 0)

    def parse_with_usage_tracking(self, text: str, token_usage) -> APIResponse:
        """Parse with token usage tracking and persistence"""
        result = self.parse_with_retry(text, token_usage)

        if result.status == 'success':
            # Save successful parsing to persistent storage
            session_id = f"session_{result.data.get('total_tokens', 0)}"
            self.persistence.save_usage(token_usage, session_id)
            logger.info(f"Saved token usage to persistent storage: {session_id}")

        return result

    def get_token_usage_summary(self) -> str:
        """Get token usage summary from persistent storage"""
        return self.persistence.get_usage_summary()
