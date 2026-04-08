"""Batch processing service for coordinating multiple receipt processing operations"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from uuid import UUID

from contracts.interfaces import (
    ImageProcessingInterface,
    ReceiptParsingInterface,
    BatchProcessingInterface,
    TokenUsageInterface
)
from tracking import TokenUsage
from api_response import APIResponse
from core.file_operations import get_image_files
from core.logging import get_batch_logger
from utils.output_formatter import format_receipt_result

# Reduce logging noise - set specific loggers to WARNING level
logging.getLogger('services.ocr_service').setLevel(logging.WARNING)
logging.getLogger('domain.validation.validation_service').setLevel(logging.WARNING)
logging.getLogger('services.retry_service').setLevel(logging.WARNING)

logger = get_batch_logger(__name__)


class BatchProcessingService(BatchProcessingInterface):
    """Service for coordinating batch processing operations"""

    def __init__(self):
        self.logger = logger

    def process_batch(self, image_files: List[Path],
                     image_processor: ImageProcessingInterface,
                     receipt_parser: ReceiptParsingInterface) -> Tuple[int, int, TokenUsage]:
        """Process multiple images and return success/failure counts and token usage"""
        successful_processes = 0
        failed_processes = 0
        total_token_usage = TokenUsage()

        for i, image_path in enumerate(image_files, 1):
            # Process image using interfaces with validation-driven retry
            ocr_text = image_processor.extract_text(str(image_path))
            result = receipt_parser.parse_with_validation_driven_retry(ocr_text, str(image_path))

            # Collect structured result for formatted output
            # result is now ParsingResult with .parsed, .valid, .error, .token_usage
            # parsed can be a Receipt object or a dict (when validation failed but data preserved)
            if result.parsed is None:
                parsed_data = {}
            elif hasattr(result.parsed, '__dict__'):
                # It's a Receipt object
                parsed_data = result.parsed.__dict__
            elif isinstance(result.parsed, dict):
                # It's a dict (preserved after validation failure)
                parsed_data = result.parsed
            else:
                parsed_data = {}

            # Extract token usage from result (accumulated across all attempts)
            token_data = {
                'input_tokens': result.token_usage.input_tokens if result.token_usage else 0,
                'output_tokens': result.token_usage.output_tokens if result.token_usage else 0,
                'total_tokens': result.token_usage.get_total_tokens() if result.token_usage else 0
            }

            # Determine actual success status:
            # SUCCESS only if validation passed (valid=True) AND receipt has meaningful content
            has_meaningful_data = (
                result.valid and
                parsed_data and
                parsed_data.get('items') and
                len(parsed_data.get('items', [])) > 0 and
                parsed_data.get('total') is not None
            )

            formatted_result = {
                'image_path': str(image_path),
                'success': has_meaningful_data,
                'parsed_receipt': parsed_data,  # Preserved even on validation failure
                'retries': receipt_parser.get_current_retries() if hasattr(receipt_parser, 'get_current_retries') else [],
                'validation_error': result.error if result.error else None,
                'token_usage': token_data
            }

            # Print formatted output instead of raw logging
            print(format_receipt_result(formatted_result))

            # Track token usage from result (accumulated across initial + all retries)
            if result.token_usage:
                total_token_usage.add_usage(
                    result.token_usage.input_tokens,
                    result.token_usage.output_tokens
                )

            # Update counters - success only if validation passed
            if result.valid and parsed_data:
                successful_processes += 1
            else:
                failed_processes += 1

        return successful_processes, failed_processes, total_token_usage

    def validate_image_files(self, imgs_dir: Path) -> List[Path]:
        """Validate and get list of image files to process"""
        if not imgs_dir.exists():
            logger.error(f"Directory 'imgs' not found: {imgs_dir}")
            return []

        image_files = get_image_files(imgs_dir)

        if not image_files:
            logger.warning(f"No image files found in {imgs_dir}")
            return []

        logger.info(f"Found {len(image_files)} image files to process")
        return image_files

    def get_batch_summary(self, successful: int, failed: int, total: int) -> str:
        """Generate batch processing summary"""
        summary = f"""
==================================================
BATCH PROCESSING COMPLETE
Successful: {successful}
Failed: {failed}
Total: {total}
Success Rate: {(successful/total*100):.1f}%
==================================================
"""
        return summary

    def print_batch_summary(self, successful: int, failed: int, total: int):
        """Print batch processing summary"""
        logger.info("=" * 50)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total: {total}")
        logger.info(f"Success Rate: {(successful/total*100):.1f}%")
        logger.info("=" * 50)

    def print_processing_result(self, result: APIResponse, image_files: List[Path], index: int):
        """Print the result of processing a single image"""
        logger.info(f"Processing image: {image_files[index]}")

        if result.status == 'success':
            logger.info("SUCCESS")
            if result.data:
                logger.info(f"Parsed Receipt: {result.data}")
        else:
            logger.error(f"FAILED: {result.error}")
            if result.error:
                logger.error(f"Error: {result.error}")
