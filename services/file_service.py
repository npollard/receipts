"""File handling service for managing file operations and result formatting"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from api_response import APIResponse
from core.file_operations import get_image_files

logger = logging.getLogger(__name__)


class FileHandlingService:
    """Service for managing file operations and result formatting"""

    def __init__(self):
        self.logger = logger

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

    def process_single_image_result(self, result: APIResponse, image_path: Path) -> APIResponse:
        """Process and format single image result"""
        self.logger.info(f"Processing image: {image_path}")

        if result.status == 'success':
            self.logger.info("Extracted Text:")
            # Handle both dict and ReceiptModel data
            if hasattr(result.data, 'model_dump'):
                # It's a Pydantic model
                data_dict = result.data.model_dump()
            else:
                # It's already a dict
                data_dict = result.data or {}

            # Print first 500 characters of extracted text
            extracted_text = data_dict.get('extracted_text', '')
            if extracted_text:
                preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                self.logger.info(preview)

            self.logger.info("SUCCESS")

            # Print parsed receipt if available
            if 'parsed_receipt' in data_dict:
                self.logger.info(f"Parsed Receipt: {data_dict['parsed_receipt']}")
            elif hasattr(result.data, 'model_dump'):
                # If it's a ReceiptModel, show the model data
                self.logger.info(f"Parsed Receipt: {data_dict}")
        else:
            self.logger.error("FAILED")
            if result.error:
                self.logger.error(f"Error: {result.error}")

        return result

    def format_result_data(self, result: APIResponse) -> Dict[str, Any]:
        """Format result data for consistent output"""
        if not result.data:
            return {}

        # Handle both dict and ReceiptModel data
        if hasattr(result.data, 'model_dump'):
            # It's a Pydantic model
            return result.data.model_dump()
        else:
            # It's already a dict
            return result.data or {}

    def enrich_result_with_metadata(self, result: APIResponse, metadata: Dict[str, Any]) -> APIResponse:
        """Enrich result with additional metadata"""
        if result.status == 'success' and result.data:
            result_data = self.format_result_data(result)
            result_data.update(metadata)
            return APIResponse.success(result_data)
        return result
