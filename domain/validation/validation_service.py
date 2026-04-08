"""Validation service for receipt data validation"""

import json
from decimal import Decimal
from pydantic import ValidationError

from models.receipt import Receipt
from api_response import APIResponse
from core.logging import get_validation_logger
from core.exceptions import ValidationError as CustomValidationError

logger = get_validation_logger(__name__)


class ValidationService:
    """Service for validating receipt data and responses"""

    def validate_response_content(self, response) -> APIResponse:
        """Validate and extract content from OpenAI response"""
        parsed_data = None  # Track parsed data for preservation on failure

        try:
            # Log raw response content for debugging
            logger.debug(f"Raw response content: {repr(response.content)}")
            logger.debug(f"Response type: {type(response.content)}")
            logger.debug(f"Response length: {len(response.content)}")

            # Check if response content exists
            if not hasattr(response, 'content') or not response.content:
                return APIResponse.failure("Empty response content received from AI model", data=None)

            # Parse JSON content
            try:
                parsed_data = json.loads(response.content)
            except json.JSONDecodeError as e:
                return APIResponse.failure(f"Invalid JSON format in response: {str(e)}. Response content: {response.content[:200]}...", data=None)
            except Exception as e:
                return APIResponse.failure(f"Unexpected error parsing JSON response: {str(e)}", data=None)

            # Validate parsed data structure
            if not isinstance(parsed_data, dict):
                return APIResponse.failure(f"Expected JSON object, got {type(parsed_data).__name__}", data=parsed_data)

            # Check for required fields
            required_fields = ['date', 'total', 'items']
            missing_fields = [field for field in required_fields if field not in parsed_data]
            if missing_fields:
                return APIResponse.failure(f"Missing required fields in response: {missing_fields}", data=parsed_data)

            # Validate that receipt has meaningful content (at least 1 item)
            if 'items' in parsed_data and parsed_data['items'] is not None:
                if not isinstance(parsed_data['items'], list):
                    return APIResponse.failure(f"Expected items to be a list, got {type(parsed_data['items']).__name__}", data=parsed_data)

                # Receipts with 0 items are invalid
                if len(parsed_data['items']) == 0:
                    return APIResponse.failure("Receipt must contain at least 1 item to be valid", data=parsed_data)

            # Validate data types
            if 'total' in parsed_data and parsed_data['total'] is not None:
                try:
                    # Handle different numeric formats
                    if isinstance(parsed_data['total'], str):
                        parsed_data['total'] = Decimal(parsed_data['total'].replace(',', '.'))
                    elif isinstance(parsed_data['total'], (int, float)):
                        parsed_data['total'] = Decimal(str(parsed_data['total']))
                    elif not isinstance(parsed_data['total'], Decimal):
                        raise ValueError(f"Invalid total type: {type(parsed_data['total']).__name__}")
                except (ValueError, TypeError) as e:
                    return APIResponse.failure(f"Invalid total value '{parsed_data['total']}': {str(e)}", data=parsed_data)

            if 'items' in parsed_data and parsed_data['items'] is not None:
                if not isinstance(parsed_data['items'], list):
                    return APIResponse.failure(f"Expected items to be a list, got {type(parsed_data['items']).__name__}", data=parsed_data)

                # Validate each item
                for i, item in enumerate(parsed_data['items']):
                    if not isinstance(item, dict):
                        return APIResponse.failure(f"Item {i+1} should be a dictionary, got {type(item).__name__}", data=parsed_data)

                    if 'name' in item and not isinstance(item['name'], str):
                        return APIResponse.failure(f"Item {i+1} name should be a string, got {type(item['name']).__name__}", data=parsed_data)

                    if 'price' in item and item['price'] is not None:
                        try:
                            if isinstance(item['price'], str):
                                item['price'] = Decimal(item['price'].replace(',', '.'))
                            elif isinstance(item['price'], (int, float)):
                                item['price'] = Decimal(str(item['price']))
                            elif not isinstance(item['price'], Decimal):
                                raise ValueError(f"Invalid price type: {type(item['price']).__name__}")
                        except (ValueError, TypeError) as e:
                            return APIResponse.failure(f"Invalid price value '{item['price']}' in item {i+1}: {str(e)}", data=parsed_data)

            logger.debug(f"Parsed JSON data: {parsed_data}")
            logger.info(f"Successfully validated receipt with {len(parsed_data.get('items', []))} items")

            # Use the graceful Pydantic validation with total adjustment
            return self.validate_with_pydantic(parsed_data, 0, 0)

        except Exception as e:
            logger.error(f"Unexpected error during response validation: {str(e)}")
            return APIResponse.failure(f"Unexpected error during response validation: {str(e)}", data=parsed_data)

    def validate_with_pydantic(self, parsed_data: dict, input_tokens: int, output_tokens: int) -> APIResponse:
        """Validate parsed data with Pydantic and handle errors"""
        try:
            validated = Receipt.model_validate(parsed_data)
            logger.info(f"Successfully validated receipt with {len(validated.items)} items")

            # Return the actual ReceiptModel, not a dict
            return APIResponse.success(validated)

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

            # Return validation failure WITH the parsed data for preservation
            logger.error(f"Pydantic validation failed: {str(e)}")
            logger.info(f"Preserving parsed data despite validation failure: {parsed_data}")
            return APIResponse.failure(f"Data validation failed: {str(e)}", data=parsed_data)
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            return APIResponse.failure(f"Validation error: {str(e)}", data=parsed_data)

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
