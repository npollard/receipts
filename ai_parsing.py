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
        self._system_prompt = "You are a receipt parser that returns valid JSON only."
        logger.info(f"Initialized ReceiptParser with model: {model_name}")

    def parse(self, text: str) -> APIResponse:
        """Parse OCR text into structured receipt data"""
        logger.debug(f"Parsing OCR text of length: {len(text)}")
        logger.debug(f"OCR text content: {repr(text[:200])}...")  # Log raw text for debugging

        prompt = self._build_prompt(text)
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            response = self.llm.invoke(messages)

            # Log the raw response content for debugging
            logger.debug(f"Raw response content: {repr(response.content)}")
            logger.debug(f"Response type: {type(response.content)}")
            logger.debug(f"Response length: {len(response.content)}")

            # Extract token usage from response
            usage_data = getattr(response, 'usage_metadata', {})
            input_tokens = usage_data.get('input_tokens', 0)
            output_tokens = usage_data.get('output_tokens', 0)

            logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")

            # Check if response content is empty or invalid
            if not response.content or not response.content.strip():
                logger.error("Empty response from OpenAI")
                return APIResponse.failure("Empty response from AI model")

            # Try to parse JSON
            try:
                parsed_data = json.loads(response.content)
                logger.debug(f"Parsed JSON data: {parsed_data}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                logger.error(f"Response content that failed: {repr(response.content)}")
                return APIResponse.failure(f"Failed to parse JSON: {str(e)}")

            # Validate with Pydantic v2
            try:
                validated = Receipt.model_validate(parsed_data)
                result = validated.model_dump()
                logger.info(f"Successfully validated receipt with {len(result.get('items', []))} items")

                # Convert Decimal objects to floats for JSON serialization
                if 'items' in result:
                    for item in result['items']:
                        if 'price' in item and isinstance(item['price'], Decimal):
                            item['price'] = float(item['price'])

                if 'total' in result and isinstance(result['total'], Decimal):
                    result['total'] = float(result['total'])

                # Add token usage to response data
                result["_token_usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }

                return APIResponse.success(result)

            except ValidationError as e:
                logger.warning(f"Validation failed: {e.errors()}")

                # Convert validation errors to JSON-serializable format using Pydantic v2 methods
                validation_error = {
                    "error": "validation_failed",
                    "details": e.errors(),  # Pydantic v2 errors are already JSON-serializable
                    "raw": parsed_data
                }

                # Convert Decimal objects in raw data to floats for JSON serialization
                if 'items' in parsed_data:
                    for item in parsed_data['items']:
                        if 'price' in item and isinstance(item['price'], Decimal):
                            item['price'] = float(item['price'])

                if 'total' in parsed_data and isinstance(parsed_data['total'], Decimal):
                    parsed_data['total'] = float(parsed_data['total'])

                # Add token usage to validation error
                validation_error["_token_usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }

                # Return FAILURE for validation errors, not success
                return APIResponse.failure(f"Validation failed: {len(e.errors())} errors found")

        except Exception as e:
            logger.error(f"Unexpected parsing error: {str(e)}")
            return APIResponse.failure(f"Parsing error: {str(e)}")

    def parse_with_usage_tracking(self, text: str, token_usage) -> APIResponse:
        """Parse with token usage tracking"""
        result = self.parse(text)

        # Update token usage tracker if successful
        if result.status == "success" and "_token_usage" in result.data:
            token_usage.add_usage(
                result.data["_token_usage"]["input_tokens"],
                result.data["_token_usage"]["output_tokens"]
            )

        return result

    def _build_prompt(self, ocr_text: str) -> str:
        """Build the parsing prompt"""
        return f"""
You are a receipt parser.

Given the OCR text below, extract:
- Date
- Items (name, category, price paid)
- Total

IMPORTANT: Return ONLY valid JSON. No explanations, no markdown, no code blocks.
If you cannot extract valid data, return:
{{"date": null, "items": [], "total": null}}

OCR TEXT:
{ocr_text}
"""
