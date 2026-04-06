"""Validation utilities for receipt parsing"""

import logging
from typing import Dict, Any
import json
from decimal import Decimal
from pydantic import ValidationError

from models import Receipt
from api_response import APIResponse

logger = logging.getLogger(__name__)


def validate_response_content(response) -> APIResponse:
    """Validate and extract content from OpenAI response"""
    # Log raw response content for debugging
    logger.debug(f"Raw response content: {repr(response.content)}")
    logger.debug(f"Response type: {type(response.content)}")
    logger.debug(f"Response length: {len(response.content)}")

    # Try to parse JSON
    try:
        parsed_data = json.loads(response.content)
        logger.debug(f"Parsed JSON data: {parsed_data}")

        # Additional validation: check if it's a dict and has expected structure
        if not isinstance(parsed_data, dict):
            logger.error(f"Expected dict, got {type(parsed_data)}: {parsed_data}")
            return APIResponse.failure(f"Expected JSON object, got {type(parsed_data)}")

        return APIResponse.success(parsed_data)

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        logger.error(f"Response content that failed: {repr(response.content)}")
        return APIResponse.failure(f"Failed to parse JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error parsing response: {str(e)}")
        if "JSONDecodeError" in str(e):
            return APIResponse.failure(f"Failed to parse JSON: {str(e)}")
        else:
            return APIResponse.failure(f"Unexpected parsing error: {str(e)}")


def validate_with_pydantic(parsed_data: dict, input_tokens: int, output_tokens: int) -> APIResponse:
    """Validate parsed data with Pydantic and handle errors"""
    try:
        validated = Receipt.model_validate(parsed_data)
        result = validated.model_dump()
        logger.info(f"Successfully validated receipt with {len(result.get('items', []))} items")

        # Convert Decimal objects to floats for JSON serialization
        result = _convert_decimals_to_floats(result)

        # Add token usage to response data
        result["_token_usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

        return APIResponse.success(result)

    except ValidationError as e:
        # Log detailed validation errors
        error_details = e.errors()
        logger.error(f"Validation failed with {len(error_details)} errors:")
        for i, error in enumerate(error_details, 1):
            loc = error.get('loc', ['unknown'])
            field = loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else (str(loc) if loc else 'unknown')
            message = error.get('msg', 'unknown error')
            error_type = error.get('type', 'unknown')
            logger.error(f"  Error {i}: Field '{field}' - {message} (type: {error_type})")

        # Log the problematic data for debugging
        logger.debug(f"Problematic data: {json.dumps(parsed_data, indent=2, default=str)}")

        return handle_validation_error(e, parsed_data, input_tokens, output_tokens)


def handle_validation_error(e: ValidationError, parsed_data: dict, input_tokens: int, output_tokens: int) -> APIResponse:
    """Handle Pydantic validation errors with proper serialization"""
    # Convert validation errors to JSON-serializable format using Pydantic v2 methods
    error_details = e.errors()

    # Create detailed error message
    error_messages = []
    for error in error_details:
        loc = error.get('loc', ['unknown'])
        field = loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else (str(loc) if loc else 'unknown')
        message = error.get('msg', 'unknown error')
        error_type = error.get('type', 'unknown')
        error_messages.append(f"Field '{field}': {message} ({error_type})")

    detailed_error_msg = f"Validation failed: {'; '.join(error_messages)}"

    validation_error = {
        "error": "validation_failed",
        "details": error_details,  # Pydantic v2 errors are already JSON-serializable
        "error_messages": error_messages,
        "raw": parsed_data or {}
    }

    # Convert Decimal objects in raw data to floats for JSON serialization
    if parsed_data:
        parsed_data = _convert_decimals_to_floats(parsed_data)

    # Add token usage to validation error
    validation_error["_token_usage"] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
    }

    # Return FAILURE for validation errors, not success
    return APIResponse.failure(detailed_error_msg)


def _convert_decimals_to_floats(data: dict) -> dict:
    """Convert Decimal objects to floats for JSON serialization"""
    if 'items' in data:
        for item in data['items']:
            if 'price' in item and isinstance(item['price'], Decimal):
                item['price'] = float(item['price'])

    if 'total' in data and isinstance(data['total'], Decimal):
        data['total'] = float(data['total'])

    return data
