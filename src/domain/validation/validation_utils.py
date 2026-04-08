"""Validation utilities for receipt parsing"""

import logging
from typing import Dict, Any
import json
from decimal import Decimal
from pydantic import ValidationError

from models.receipt import Receipt
from api_response import APIResponse
from .validation_service import ValidationService

logger = logging.getLogger(__name__)


# Create a global instance for backward compatibility
_validation_service = ValidationService()


def validate_response_content(response) -> APIResponse:
    """Validate and extract content from OpenAI response"""
    return _validation_service.validate_response_content(response)


def validate_with_pydantic(parsed_data: dict, input_tokens: int, output_tokens: int) -> APIResponse:
    """Validate parsed data with Pydantic and handle errors"""
    return _validation_service.validate_with_pydantic(parsed_data, input_tokens, output_tokens)


def handle_validation_error(e: ValidationError, parsed_data: dict, input_tokens: int, output_tokens: int) -> APIResponse:
    """Handle Pydantic validation errors with proper serialization"""
    return _validation_service.handle_validation_error(e, parsed_data, input_tokens, output_tokens)


def _convert_decimals_to_floats(data: dict) -> dict:
    """Convert Decimal objects to floats for JSON serialization"""
    return _validation_service._convert_decimals_to_floats(data)
