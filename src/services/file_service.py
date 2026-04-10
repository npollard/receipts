"""File handling service for managing file operations and result formatting"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from contracts.interfaces import FileHandlingInterface
from api_response import APIResponse
from core.file_operations import get_image_files

logger = logging.getLogger(__name__)


class FileHandlingService(FileHandlingInterface):
    """Service for managing file operations and result formatting"""

    def __init__(self):
        self.logger = logger

    def validate_file(self, file_path: Path) -> bool:
        """Validate file exists and is readable"""
        return file_path.exists() and file_path.is_file()

    def format_result(self, result: APIResponse, file_path: Path) -> Dict[str, Any]:
        """Format processing result with file metadata"""
        formatted_result = {
            "status": result.status,
            "data": result.data,
            "error": result.error,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size if self.validate_file(file_path) else None
        }

        if result.error:
            formatted_result["error"] = result.error

        return formatted_result

    def get_image_files(self, directory: Path) -> List[Path]:
        """Get list of image files from directory"""
        return get_image_files(directory)

    def enrich_result_with_metadata(self, result: APIResponse, metadata: Dict[str, Any]) -> APIResponse:
        """Enrich API response with additional metadata"""
        enriched_data = result.data.copy() if result.data else {}
        enriched_data.update(metadata)

        return APIResponse(
            status=result.status,
            data=enriched_data,
            error=result.error
        )

    def validate_and_get_image_files(self, imgs_dir: Path) -> List[Path]:
        """Validate and get list of image files to process"""
        if not imgs_dir.exists():
            self.logger.error(f"Directory 'imgs' not found: {imgs_dir}")
            return []

        image_files = self.get_image_files(imgs_dir)

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

    def _convert_data_to_dict(self, data: Any) -> Dict[str, Any]:
        """Convert data to dict - handles both Pydantic models and dicts"""
        if data is None:
            return {}
        if isinstance(data, dict):
            return data
        # Assume Pydantic model with model_dump()
        return data.model_dump()

    def process_single_image_result(self, result: APIResponse, image_path: Path) -> APIResponse:
        """Process and format single image result"""
        self.logger.info(f"Processing image: {image_path}")

        if result.status == 'success':
            self.logger.info("Extracted Text:")
            data_dict = self._convert_data_to_dict(result.data)

            # Print first 500 characters of extracted text
            extracted_text = data_dict.get('extracted_text', '')
            if extracted_text:
                preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                self.logger.info(preview)

            self.logger.info("SUCCESS")

            # Print parsed receipt if available
            if 'parsed_receipt' in data_dict:
                self.logger.info(f"Parsed Receipt: {data_dict['parsed_receipt']}")
            elif data_dict:
                self.logger.info(f"Parsed Receipt: {data_dict}")
        else:
            self.logger.error("FAILED")
            if result.error:
                self.logger.error(f"Error: {result.error}")

        return result

    def format_result_data(self, result: APIResponse) -> Dict[str, Any]:
        """Format result data for consistent output"""
        return self._convert_data_to_dict(result.data)

    def enrich_result_with_metadata(self, result: APIResponse, metadata: Dict[str, Any]) -> APIResponse:
        """Enrich result with additional metadata"""
        if result.status == 'success' and result.data:
            result_data = self.format_result_data(result)
            result_data.update(metadata)
            return APIResponse.success(result_data)
        return result
