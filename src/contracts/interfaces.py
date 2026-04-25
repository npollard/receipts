"""Interface contracts for clean separation between layers"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from pathlib import Path

from api_response import APIResponse
from tracking import TokenUsage

if TYPE_CHECKING:
    from domain.parsing.parsing_result import ParsingResult
    from services.batch_service import BatchObservability


class ImageProcessingInterface(ABC):
    """Interface for image processing operations"""

    @abstractmethod
    def preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """Preprocess image for OCR"""
        pass

    @abstractmethod
    def extract_text(self, image_path: str, use_vision_fallback: bool = False) -> str:
        """Extract text from image"""
        pass

    @abstractmethod
    def score_ocr_quality(self, text: str) -> float:
        """Score OCR text quality (0-1)"""
        pass


class ReceiptParsingInterface(ABC):
    """Interface for receipt parsing operations"""

    @abstractmethod
    def parse_text(self, text: str) -> "ParsingResult":
        """Parse receipt text into structured data"""
        pass

    @abstractmethod
    def get_token_usage(self) -> TokenUsage:
        """Get token usage from parsing operations"""
        pass

    @abstractmethod
    def get_current_retries(self) -> List[str]:
        """Get list of retry strategies used in current parsing session"""
        pass


class LanguageModelInterface(ABC):
    """Interface for chat model invocation."""

    @abstractmethod
    def invoke(self, messages: List[Any]) -> Any:
        """Invoke the model with chat messages."""
        pass


class BatchProcessingInterface(ABC):
    """Interface for batch processing operations"""

    @abstractmethod
    def process_batch(self, image_files: List[Path],
                     image_processor: ImageProcessingInterface,
                     receipt_parser: ReceiptParsingInterface) -> Tuple[int, int, TokenUsage, Optional["BatchObservability"]]:
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
