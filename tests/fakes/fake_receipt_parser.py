"""Fake ReceiptParser for fast testing without LLM calls"""
from typing import Dict, Any, Optional

from contracts.interfaces import ReceiptParsingInterface
from tracking import TokenUsage
from domain.parsing.receipt_parser import ParsingResult


class FakeReceiptParser(ReceiptParsingInterface):
    """Fake receipt parser that returns pre-configured results without LLM"""

    def __init__(self, result: Dict[str, Any] = None, fail: bool = False):
        """
        Args:
            result: The dict to return as parsed data (should include _token_usage)
            fail: If True, returns a failed ParsingResult
        """
        self.result = result or {
            "date": "2026-03-30",
            "total": 0.0,
            "items": [],
            "_token_usage": {
                "input_tokens": 0,
                "output_tokens": 0
            }
        }
        self.fail = fail
        self.call_count = 0

    def parse_text(self, text: str) -> ParsingResult:
        """Fake text parsing"""
        self.call_count += 1
        result = ParsingResult()

        if self.fail:
            result.parsed = None
            result.valid = False
            result.error = "Validation failed"
        else:
            # Copy result without _token_usage for parsed data
            parsed_data = {k: v for k, v in self.result.items() if k != "_token_usage"}
            result.parsed = parsed_data
            result.valid = True
            result.error = None

            # Add token usage if provided
            token_usage = self.result.get("_token_usage", {})
            result.token_usage.add_usage(
                token_usage.get("input_tokens", 0),
                token_usage.get("output_tokens", 0)
            )

        return result

    def parse_with_usage_tracking(self, text: str, token_usage: TokenUsage = None) -> ParsingResult:
        """Fake parsing with usage tracking"""
        return self.parse_text(text)

    def parse_with_validation_driven_retry(self, ocr_text: str, image_path: str = None) -> ParsingResult:
        """Fake parsing with retry logic"""
        return self.parse_text(ocr_text)

    def get_token_usage(self) -> TokenUsage:
        """Return fake TokenUsage object"""
        token_usage = self.result.get("_token_usage", {})
        tu = TokenUsage()
        tu.add_usage(
            token_usage.get("input_tokens", 0),
            token_usage.get("output_tokens", 0)
        )
        return tu

    def get_token_usage_safely(self) -> Dict[str, Any]:
        """Return fake token usage"""
        token_usage = self.result.get("_token_usage", {})
        return {
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "total_tokens": token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0),
            "estimated_cost": 0.0
        }

    def get_current_retries(self) -> list:
        """Return list of retry strategies used"""
        return []
