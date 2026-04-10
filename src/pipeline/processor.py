"""Pure orchestrator pipeline for receipt processing"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from contracts.interfaces import (
    ImageProcessingInterface,
    ReceiptParsingInterface,
)
from tracking import TokenUsage
from api_response import APIResponse
from image_processing import VisionProcessor
from domain.parsing.receipt_parser import ReceiptParser

from services.batch_service import BatchProcessingService
from services.file_service import FileHandlingService
from services.token_service import TokenUsageService
from core.logging import get_pipeline_logger
from core.exceptions import OCRError, ParsingError, ReceiptProcessingError

logger = get_pipeline_logger(__name__)


class ReceiptProcessor:
    """Pure orchestrator that coordinates receipt processing through clean interfaces.

    This class has a single responsibility: orchestrate OCR and parsing.
    No persistence, no business logic - pure orchestration only.
    """

    def __init__(
        self,
        image_processor: Optional[ImageProcessingInterface] = None,
        receipt_parser: Optional[ReceiptParsingInterface] = None,
        db_manager=None
    ):
        # Initialize core interfaces (dependency injection with defaults)
        self.image_processor = image_processor or VisionProcessor()
        self.receipt_parser = receipt_parser or ReceiptParser()
        self.ai_parser = self.receipt_parser  # Temporary alias for test compatibility

        # Initialize token usage tracking
        self.token_usage = TokenUsage()

        # Initialize orchestration services
        self.batch_service = BatchProcessingService()
        self.file_service = FileHandlingService()
        self.token_service = TokenUsageService()

        # Database and persistence (for backward compatibility)
        self.db_manager = db_manager
        self.user_manager = None
        self.repository = None
        self.persistence = None
        self._current_user_id = "default"

        logger.info("Initialized ReceiptProcessor (pure orchestrator)")

    def process_image(self, image_path: str) -> APIResponse:
        """Process a single image through OCR and parsing.

        This is a pure orchestration method with single execution flow:
        1. Extract OCR text
        2. Parse receipt data
        3. Return result

        No persistence, no business logic - pure orchestration only.
        """
        logger.info(f"Processing image: {image_path}")

        try:
            # Step 1: Extract OCR text using interface
            ocr_text = self.image_processor.extract_text(image_path)
            logger.info(f"Extracted OCR text: {ocr_text[:100]}..." if len(ocr_text) > 100 else f"Extracted OCR text: {ocr_text}")

            # Step 2: Parse with AI using validation-driven retry
            parse_result = self.receipt_parser.parse_with_validation_driven_retry(ocr_text, image_path)

            # Aggregate token usage from parsing result
            if parse_result.token_usage:
                self.token_usage.add_usage(
                    parse_result.token_usage.input_tokens,
                    parse_result.token_usage.output_tokens
                )

            # Step 3: Format and return result
            formatted = self.file_service.format_result(
                APIResponse.success({"parsed": parse_result.parsed, "ocr_text": ocr_text}),
                Path(image_path)
            )

            if parse_result.valid and parse_result.parsed:
                parsed_data = parse_result.parsed.copy()
                token_usage = parse_result.token_usage
                parsed_data["_token_usage"] = {
                    "input_tokens": token_usage.input_tokens,
                    "output_tokens": token_usage.output_tokens,
                    "total_tokens": token_usage.input_tokens + token_usage.output_tokens
                }

                return APIResponse.success({
                    "image_path": image_path,
                    "ocr_text": ocr_text,
                    "parsed_receipt": parsed_data
                })
            elif not parse_result.valid:
                return APIResponse.failure(parse_result.error or "Parsing failed")
            else:
                return APIResponse.failure("No receipt data extracted")

        except (OCRError, ParsingError) as e:
            logger.error(f"Pipeline error processing {image_path}: {str(e)}")
            return APIResponse.failure(f"Processing error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing {image_path}: {str(e)}")
            return APIResponse.failure(f"Unexpected error: {str(e)}")

    def process_directly(self, image_path: str) -> APIResponse:
        """Process directly (alias for process_image)"""
        return self.process_image(image_path)

    def get_token_usage_summary(self) -> str:
        """Get current token usage summary"""
        return self.token_usage.get_summary()

    def reset_token_usage(self):
        """Reset token usage tracking"""
        self.token_usage = TokenUsage()

    def get_current_user(self) -> str:
        """Get current user ID"""
        return self._current_user_id

    def switch_user(self, user_id: str):
        """Switch to a different user context"""
        self._current_user_id = user_id

    def get_user_context(self) -> dict:
        """Get current user context"""
        return {
            'user_id': self._current_user_id,
            'email': f'{self._current_user_id}@example.com'
        }


def process_receipt(image_path: str, **kwargs) -> APIResponse:
    """
    Backward-compatible wrapper for ReceiptProcessor.
    DO NOT add logic here — delegate only.
    """
    processor = ReceiptProcessor(**kwargs)
    return processor.process_image(image_path)


# Pure orchestration functions that delegate to services
def process_single_image(image_path: Path, processor: ReceiptProcessor) -> APIResponse:
    """Process a single image file"""
    result = processor.process_image(str(image_path))
    return processor.file_service.process_single_image_result(result, image_path)


def validate_and_get_image_files(imgs_dir: Path) -> List[Path]:
    """Validate and get list of image files to process"""
    file_service = FileHandlingService()
    return file_service.validate_and_get_image_files(imgs_dir)


def print_batch_summary(successful: int, failed: int, total: int):
    """Print batch processing summary"""
    batch_service = BatchProcessingService()
    batch_service.print_batch_summary(successful, failed, total)


def print_processing_result(result: APIResponse, image_files: List[Path], index: int):
    """Print the result of processing a single image"""
    file_service = FileHandlingService()
    file_service.print_processing_result(result, image_files, index)


def print_token_usage_summary(token_usage):
    """Print token usage summary for current batch"""
    from services.token_service import TokenUsageService
    service = TokenUsageService()
    print(f"Token Usage - Input: {token_usage.input_tokens}, "
          f"Output: {token_usage.output_tokens}, "
          f"Total: {token_usage.get_total_tokens()}")


def save_token_usage_to_persistence(token_usage, token_service=None):
    """Save token usage to persistence layer"""
    from services.token_service import TokenUsageService
    from uuid import UUID
    if token_service is None:
        token_service = TokenUsageService()
    # Delegate to service - persistence moved to application layer
    # This function remains for backward compatibility
    token_service.save_token_usage_to_persistence(
        user_id=UUID(int=0),  # Default system user
        token_usage=token_usage
    )


def print_usage_summary(show_persisted: bool = False, token_service=None):
    """Print usage summary (backward-compatible adapter)"""
    from services.token_service import TokenUsageService
    if token_service is None:
        token_service = TokenUsageService()
    token_service.print_usage_summary(show_persisted=show_persisted)
