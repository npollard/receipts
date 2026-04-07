"""Core receipt parsing functionality"""

import logging
from typing import Dict, Any
import json
import os
from decimal import Decimal
from langchain_core.tools import tool
from services.parser_service import ParserService
from services.validation import ValidationService
from tracking import TokenUsage

logger = logging.getLogger(__name__)


class ReceiptParser:
    """Concrete implementation of receipt parsing using Parser Service"""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.parser_service = ParserService(model_name, temperature)
        self.validation_service = ValidationService()
        self.token_usage = TokenUsage()
        logger.info(f"Initialized ReceiptParser with model: {model_name}")

    def parse_text(self, text: str) -> APIResponse:
        """Main parsing method for OCR text into structured receipt data"""
        return self.parser_service.parse_text(text)

    def parse_with_retry(self, text: str, max_retries: int = 3) -> APIResponse:
        """Parse with retry logic and error fixing"""
        return self.parser_service.parse_with_retry(text, self.token_usage, max_retries)

    def parse_with_usage_tracking(self, text: str, token_usage=None) -> APIResponse:
        """Parse text with token usage tracking"""
        if token_usage is None:
            token_usage = self.token_usage
        return self.parser_service.parse_with_usage_tracking(text, token_usage)

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
