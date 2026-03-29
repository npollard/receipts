"""AI parsing utilities"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import json
import os
from decimal import Decimal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from models import Receipt
from api_response import APIResponse

logger = logging.getLogger(__name__)


class AIParser(ABC):
    """Abstract base class for AI parsing"""

    @abstractmethod
    def parse(self, text: str) -> APIResponse:
        """Parse OCR text into structured data"""
        pass


class ReceiptParser(AIParser):
    """Concrete implementation of receipt parsing using OpenAI"""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self._system_prompt = """
You are a receipt parser that returns ONLY valid JSON.

Return JSON in this exact schema:
{
  "date": "ISO-8601 date string or null",
  "total": number or null,
  "items": [
    {
      "name": "item name",
      "category": "category",
      "price": number
    }
  ]
}

CRITICAL RULES:
- Return ONLY the JSON object
- No explanations, no markdown, no code blocks
- All prices must be numbers
- If you cannot extract data, return: {"date": null, "total": null, "items": []}
"""
        logger.info(f"Initialized ReceiptParser with model: {model_name}")

    def _extract_token_usage(self, response) -> tuple[int, int, int]:
        """Extract token usage from OpenAI response"""
        usage_data = getattr(response, 'usage_metadata', {})
        input_tokens = usage_data.get('input_tokens', 0)
        output_tokens = usage_data.get('output_tokens', 0)
        return input_tokens, output_tokens, input_tokens + output_tokens

    def _validate_response_content(self, response) -> APIResponse:
        """Validate and extract content from OpenAI response"""
        # Check if response content is empty or invalid
        if not response.content or not response.content.strip():
            logger.error("Empty response from OpenAI")
            return APIResponse.failure("Empty response from AI model")

        # Log raw response content for debugging
        logger.debug(f"Raw response content: {repr(response.content)}")
        logger.debug(f"Response type: {type(response.content)}")
        logger.debug(f"Response length: {len(response.content)}")

        # Try to parse JSON
        try:
            parsed_data = json.loads(response.content)
            logger.debug(f"Parsed JSON data: {parsed_data}")
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Response content that failed: {repr(response.content)}")
            return APIResponse.failure(f"Failed to parse JSON: {str(e)}")

    def _validate_with_pydantic(self, parsed_data: dict, input_tokens: int, output_tokens: int) -> APIResponse:
        """Validate parsed data with Pydantic and handle errors"""
        try:
            validated = Receipt.model_validate(parsed_data)
            result = validated.model_dump()
            logger.info(f"Successfully validated receipt with {len(result.get('items', []))} items")

            # Convert Decimal objects to floats for JSON serialization
            result = self._convert_decimals_to_floats(result)

            # Add token usage to response data
            result["_token_usage"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }

            return APIResponse.success(result)

        except ValidationError as e:
            logger.warning(f"Validation failed: {e.errors()}")
            return self._handle_validation_error(e, parsed_data, input_tokens, output_tokens)

    def _convert_decimals_to_floats(self, data: dict) -> dict:
        """Convert Decimal objects to floats for JSON serialization"""
        if 'items' in data:
            for item in data['items']:
                if 'price' in item and isinstance(item['price'], Decimal):
                    item['price'] = float(item['price'])

        if 'total' in data and isinstance(data['total'], Decimal):
            data['total'] = float(data['total'])

        return data

    def _handle_validation_error(self, e: ValidationError, parsed_data: dict, input_tokens: int, output_tokens: int) -> APIResponse:
        """Handle Pydantic validation errors with proper serialization"""
        # Convert validation errors to JSON-serializable format using Pydantic v2 methods
        validation_error = {
            "error": "validation_failed",
            "details": e.errors(),  # Pydantic v2 errors are already JSON-serializable
            "raw": parsed_data
        }

        # Convert Decimal objects in raw data to floats for JSON serialization
        parsed_data = self._convert_decimals_to_floats(parsed_data)

        # Add token usage to validation error
        validation_error["_token_usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

        # Return FAILURE for validation errors, not success
        return APIResponse.failure(f"Validation failed: {len(e.errors())} errors found")

    def _create_fix_prompt(self, original_text: str, failed_response: dict) -> str:
        """Create a prompt to fix parsing errors"""
        return f"""
The previous parsing attempt failed. Fix these issues and return valid JSON:

FAILED RESPONSE:
{failed_response.get('data', '')}

ERRORS TO FIX:
1. Ensure valid JSON format
2. All prices must be numbers (not strings)
3. All items must have name, category, price
4. Date must be ISO-8601 format or null
5. Total must be a number or null

OCR TEXT TO RE-PARSE:
{original_text}

Return ONLY: corrected JSON object. No explanations.
"""

    def _attempt_fix(self, original_text: str, failed_response: dict, attempt: int, input_tokens: int, output_tokens: int) -> APIResponse:
        """Attempt to fix parsing errors with AI"""
        logger.info(f"Attempting to fix parsing errors on attempt {attempt + 2}")

        # Create fixing prompt
        fix_prompt = self._create_fix_prompt(original_text, failed_response)

        try:
            # Create fixing messages
            fix_messages = [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=fix_prompt)
            ]

            # Call AI to fix the response
            fix_response = self.llm.invoke(fix_messages)

            # Extract token usage from fix response
            fix_input_tokens, fix_output_tokens, fix_total_tokens = self._extract_token_usage(fix_response)

            logger.info(f"Fix attempt {attempt + 2} token usage - Input: {fix_input_tokens}, Output: {fix_output_tokens}")

            # Try to parse the fixed response
            try:
                fixed_data = json.loads(fix_response.content)
                logger.debug(f"Fixed response data: {fixed_data}")

                # Validate with Pydantic v2
                try:
                    validated = Receipt.model_validate(fixed_data)
                    fixed_result = validated.model_dump()

                    # Convert Decimal objects to floats for JSON serialization
                    fixed_result = self._convert_decimals_to_floats(fixed_result)

                    # Add combined token usage
                    total_input_tokens = input_tokens + fix_input_tokens
                    total_output_tokens = output_tokens + fix_output_tokens

                    fixed_result["_token_usage"] = {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_input_tokens + total_output_tokens
                    }

                    logger.info(f"Successfully fixed and validated on attempt {attempt + 2}")
                    return APIResponse.success(fixed_result)

                except ValidationError as e:
                    logger.warning(f"Fixed response still has validation errors: {e.errors()}")
                    return APIResponse.failure(f"Fix attempt {attempt + 2} failed: {len(e.errors())} validation errors")

            except json.JSONDecodeError as e:
                logger.error(f"Fix response has JSON error: {str(e)}")
                return APIResponse.failure(f"Fix attempt {attempt + 2} failed: JSON error - {str(e)}")

        except Exception as e:
            logger.error(f"Fix attempt {attempt + 2} failed: {str(e)}")
            return APIResponse.failure(f"Fix attempt {attempt + 2} failed: {str(e)}")

    def parse_with_retry(self, text: str, token_usage, max_retries: int = 3) -> APIResponse:
        """Parse with retry mechanism and agent loop for fixing errors"""
        logger.debug(f"Starting parsing with up to {max_retries} retries")

        # First attempt
        result = self.parse(text)

        if result.status == "success":
            logger.info("Parsing successful on attempt 1")
            # Update token usage tracker if successful
            if "_token_usage" in result.data:
                token_usage.add_usage(
                    result.data["_token_usage"]["input_tokens"],
                    result.data["_token_usage"]["output_tokens"]
                )
            return result

        # If parsing failed, try to fix with agent
        logger.warning(f"Parse attempt 1 failed: {result.error}")

        for attempt in range(1, max_retries):
            # If this is the last attempt, don't retry
            if attempt == max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded")
                return result

            # For retry attempts, try to fix the response
            fix_result = self._attempt_fix(text, result.data, attempt,
                                    result.data.get('_token_usage', {}).get('input_tokens', 0),
                                    result.data.get('_token_usage', {}).get('output_tokens', 0))

            if fix_result.status == "success":
                logger.info(f"Successfully fixed and validated on attempt {attempt + 1}")
                # Update token usage tracker
                if "_token_usage" in fix_result.data:
                    token_usage.add_usage(
                        fix_result.data["_token_usage"]["input_tokens"],
                        fix_result.data["_token_usage"]["output_tokens"]
                    )
                return fix_result

        # Should never reach here, but just in case
        return APIResponse.failure("Max retries exceeded with persistent errors")

    def parse_with_usage_tracking(self, text: str, token_usage) -> APIResponse:
        """Parse with token usage tracking"""
        result = self.parse_with_retry(text, token_usage)
        return result

    def _build_prompt(self, ocr_text: str) -> str:
        """Build the parsing prompt"""
        return f"""
Extract receipt data from this OCR text and return ONLY valid JSON:

OCR TEXT:
{ocr_text}

Remember: Return ONLY: JSON object, nothing else.
"""
