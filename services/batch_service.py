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
            # Process image using interfaces
            ocr_text = image_processor.extract_text(str(image_path))
            result = receipt_parser.parse_text(ocr_text)

            # Track token usage from successful results
            if result.status == 'success' and result.data:
                token_usage = receipt_parser.get_token_usage()
                if token_usage:
                    total_token_usage.add_usage(
                        token_usage.input_tokens,
                        token_usage.output_tokens
                    )

            # Update counters
            if result.status == 'success':
                successful_processes += 1
                logger.info(f"Successfully processed {image_path}")
            else:
                failed_processes += 1
                logger.error(f"Failed to process {image_path}")

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
