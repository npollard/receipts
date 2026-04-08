"""Batch processing service for coordinating multiple receipt processing operations"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from uuid import UUID

from tracking import TokenUsage
from api_response import APIResponse
from core.file_operations import get_image_files

logger = logging.getLogger(__name__)


class BatchProcessingService:
    """Service for coordinating batch processing operations"""

    def __init__(self):
        self.logger = logger

    def process_batch_images(self, image_files: List[Path], processor, token_usage_service) -> Tuple[int, int, TokenUsage]:
        """Process multiple images and return success/failure counts and token usage"""
        successful_processes = 0
        failed_processes = 0
        total_token_usage = TokenUsage()

        for i, image_path in enumerate(image_files, 1):
            result = processor.process_image(str(image_path))

            # Track token usage from successful results
            if result.status == 'success' and result.data:
                token_usage_data = token_usage_service.extract_token_usage_from_result(result)
                if token_usage_data:
                    total_token_usage.add_usage(
                        token_usage_data.get('input_tokens', 0),
                        token_usage_data.get('output_tokens', 0)
                    )

            # Update counters
            if result.status == 'success':
                successful_processes += 1
                self.logger.info("SUCCESS")
            else:
                failed_processes += 1
                self.logger.error("FAILED")

        return successful_processes, failed_processes, total_token_usage

    def validate_and_get_image_files(self, imgs_dir: Path) -> List[Path]:
        """Validate and get list of image files to process"""
        if not imgs_dir.exists():
            self.logger.error(f"Directory 'imgs' not found: {imgs_dir}")
            return []

        image_files = get_image_files(imgs_dir)

        if not image_files:
            self.logger.warning(f"No image files found in {imgs_dir}")
            return []

        self.logger.info(f"Found {len(image_files)} image files to process")
        return image_files

    def get_batch_summary(self, successful: int, failed: int, total: int) -> str:
        """Generate batch processing summary"""
        summary = f"""
==================================================
BATCH PROCESSING COMPLETE
Successful: {successful}
Failed: {failed}
Success Rate: {(successful / total * 100):.1f}%
==================================================
"""
        return summary

    def print_batch_summary(self, successful: int, failed: int, total: int):
        """Print batch processing summary"""
        self.logger.info("=" * 50)
        self.logger.info("BATCH PROCESSING COMPLETE")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Success Rate: {(successful / total * 100):.1f}%")
        self.logger.info("=" * 50)

    def print_processing_result(self, result: APIResponse, image_files: List[Path], index: int):
        """Print the result of processing a single image"""
        self.logger.info(f"Processing image: {image_files[index]}")

        if result.status == 'success':
            self.logger.info("SUCCESS")
            if result.data:
                self.logger.info(f"Parsed Receipt: {result.data}")
        else:
            self.logger.error("FAILED")
            if result.error:
                self.logger.error(f"Error: {result.error}")
