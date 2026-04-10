"""Validation service for receipt data validation"""

import json
from decimal import Decimal
from pydantic import ValidationError

from domain.models.receipt import Receipt
from api_response import APIResponse
from core.logging import get_validation_logger
from core.exceptions import ValidationError as CustomValidationError

logger = get_validation_logger(__name__)


class ValidationService:
    """Service for validating receipt data and responses"""

    def validate_response_content(self, response) -> APIResponse:
        """Validate and extract JSON content from response"""
        content = response.content.strip()

        if not content:
            return APIResponse.failure("Failed to parse JSON")

        try:
            parsed = json.loads(content)
        except Exception:
            return APIResponse.failure("Failed to parse JSON")

        # Only accept JSON objects (dict), not arrays or primitives
        if not isinstance(parsed, dict):
            return APIResponse.failure("Failed to parse JSON")

        return APIResponse.success(parsed)

    def validate_with_pydantic(self, parsed_data: dict, input_tokens: int, output_tokens: int) -> APIResponse:
        """Validate parsed data with Pydantic and handle errors"""
        try:
            validated = Receipt.model_validate(parsed_data)
            logger.info(f"Successfully validated receipt with {len(validated.items)} items")

            # Convert to dict and add token usage
            data = validated.model_dump()
            data["_token_usage"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            return APIResponse.success(data)

        except ValidationError as e:
            # Log detailed validation errors at debug level to reduce spam
            error_details = e.errors()
            logger.debug(f"Validation failed with {len(error_details)} errors:")
            for i, error in enumerate(error_details, 1):
                loc = error.get('loc', ['unknown'])
                field = loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else (str(loc) if loc else 'unknown')
                message = error.get('msg', 'unknown error')
                error_type = error.get('type', 'unknown')
                logger.debug(f"  Error {i}: Field '{field}' - {message} (type: {error_type})")

            # Return validation failure WITH the parsed data for preservation
            logger.debug(f"Validation failed: {str(e)}")
            logger.info(f"Preserving parsed data despite validation failure: {parsed_data}")
            return APIResponse.failure("Validation failed", data=parsed_data)
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            return APIResponse.failure("Validation failed", data=parsed_data)

    def handle_validation_error(self, e: ValidationError, parsed_data: dict, input_tokens: int, output_tokens: int) -> APIResponse:
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
            parsed_data = self._convert_decimals_to_floats(parsed_data)

        # Add token usage to validation error
        validation_error["_token_usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

        # Return FAILURE for validation errors, not success
        return APIResponse.failure(detailed_error_msg)

    def _convert_decimals_to_floats(self, data: dict) -> dict:
        """Convert Decimal objects to floats for JSON serialization"""
        if 'items' in data:
            for item in data['items']:
                if 'price' in item and isinstance(item['price'], Decimal):
                    item['price'] = float(item['price'])

        if 'total' in data and isinstance(data['total'], Decimal):
            data['total'] = float(data['total'])

        return data
