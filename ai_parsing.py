"""AI parsing utilities"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import json
import os
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

            # Validate with Pydantic
            try:
                validated = Receipt(**parsed_data)
                result = validated.model_dump()
                logger.info(f"Successfully validated receipt with {len(result.get('items', []))} items")

                # Add token usage to response data
                result["_token_usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }

                return APIResponse.success(result)

            except ValidationError as e:
                logger.warning(f"Validation failed: {e.errors()}")

                # Convert validation errors to JSON-serializable format
                serializable_errors = []
                for error in e.errors():
                    serializable_error = {
                        'loc': error.get('loc', []),
                        'msg': error.get('msg', ''),
                        'type': error.get('type', ''),
                        'ctx': error.get('ctx', {})
                    }
                    # Convert any non-serializable objects in ctx
                    if 'ctx' in error and error['ctx']:
                        for key, value in error['ctx'].items():
                            if hasattr(value, '__dict__'):
                                serializable_error['ctx'][key] = str(value)
                            elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                serializable_error['ctx'][key] = str(value)

                    serializable_errors.append(serializable_error)

                validation_error = {
                    "error": "validation_failed",
                    "details": serializable_errors,
                    "raw": parsed_data
                }

                # Add token usage to validation error
                validation_error["_token_usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }

                return APIResponse.success(validation_error)

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
