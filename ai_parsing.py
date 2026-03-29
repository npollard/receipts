"""AI parsing utilities"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import json
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from models import Receipt
from api_response import APIResponse


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

    def parse(self, text: str) -> APIResponse:
        """Parse OCR text into structured receipt data"""
        prompt = self._build_prompt(text)
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            response = self.llm.invoke(messages)

            # Extract token usage from response
            usage_data = getattr(response, 'usage_metadata', {})
            input_tokens = usage_data.get('input_tokens', 0)
            output_tokens = usage_data.get('output_tokens', 0)

            parsed_data = json.loads(response.content)

            # Validate with Pydantic
            try:
                validated = Receipt(**parsed_data)
                result = validated.dict()

                # Add token usage to response data
                result["_token_usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }

                return APIResponse.success(result)

            except ValidationError as e:
                validation_error = {
                    "error": "validation_failed",
                    "details": e.errors(),
                    "raw": parsed_data
                }

                # Add token usage to validation error
                validation_error["_token_usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }

                return APIResponse.success(validation_error)

        except json.JSONDecodeError as e:
            return APIResponse.failure(f"Failed to parse JSON: {str(e)}")
        except Exception as e:
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

Return valid JSON only.

OCR TEXT:
{ocr_text}
"""
