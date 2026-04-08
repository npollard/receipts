"""Interface contracts for clean separation between layers"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from uuid import UUID

from api_response import APIResponse
from tracking import TokenUsage


class ImageProcessingInterface(ABC):
    """Interface for image processing operations"""
    
    @abstractmethod
    def preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """Preprocess image for OCR"""
        pass
    
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from image"""
        pass


class ReceiptParsingInterface(ABC):
    """Interface for receipt parsing operations"""
    
    @abstractmethod
    def parse_text(self, text: str) -> APIResponse:
        """Parse receipt text into structured data"""
        pass
    
    @abstractmethod
    def get_token_usage(self) -> TokenUsage:
        """Get token usage from parsing operations"""
        pass


class BatchProcessingInterface(ABC):
    """Interface for batch processing operations"""
    
    @abstractmethod
    def process_batch(self, image_files: List[Path], 
                     image_processor: ImageProcessingInterface,
                     receipt_parser: ReceiptParsingInterface) -> Tuple[int, int, TokenUsage]:
        """Process multiple images and return results"""
        pass
    
    @abstractmethod
    def validate_image_files(self, imgs_dir: Path) -> List[Path]:
        """Validate and get list of image files"""
        pass


class TokenUsageInterface(ABC):
    """Interface for token usage tracking"""
    
    @abstractmethod
    def extract_from_result(self, result: APIResponse) -> Dict[str, Any]:
        """Extract token usage from API response"""
        pass
    
    @abstractmethod
    def aggregate_usage(self, usage_list: List[TokenUsage]) -> TokenUsage:
        """Aggregate multiple token usage objects"""
        pass


class FileHandlingInterface(ABC):
    """Interface for file operations"""
    
    @abstractmethod
    def validate_file(self, file_path: Path) -> bool:
        """Validate file exists and is readable"""
        pass
    
    @abstractmethod
    def format_result(self, result: APIResponse, file_path: Path) -> Dict[str, Any]:
        """Format processing result with file metadata"""
        pass
