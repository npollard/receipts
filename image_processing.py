"""Image processing utilities"""

import logging
from abc import ABC, abstractmethod
from typing import Any
from langchain_core.tools import tool
from services.ocr.ocr_service import OCRService

logger = logging.getLogger(__name__)


class ImageProcessor(ABC):
    """Abstract base class for image processing"""

    @abstractmethod
    def preprocess(self, image_path: str) -> Any:
        """Preprocess image for OCR"""
        pass

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from image"""
        pass


class VisionProcessor(ImageProcessor):
    """Concrete implementation of image processing using EasyOCR Service"""

    def __init__(self, use_gpu: bool = False, lang: list = ['en'], confidence_threshold: float = 0.7):
        self.ocr_service = OCRService(use_gpu=use_gpu, lang=lang, confidence_threshold=confidence_threshold)
        logger.info(f"Initialized VisionProcessor with EasyOCR (GPU: {use_gpu}, Lang: {lang}, Confidence: {confidence_threshold})")

    def preprocess(self, image_path: str) -> Any:
        """Public interface for preprocessing"""
        return self.ocr_service.preprocess_image(image_path)

    def extract_text(self, image_path: str) -> str:
        """Public interface for OCR extraction"""
        return self.ocr_service.extract_text(image_path)

    @tool
    def preprocess_image_tool(self, image_path: str) -> Any:
        """LangChain tool for image preprocessing"""
        return self.preprocess(image_path)

    @tool
    def extract_vision_text_tool(self, image_path: str) -> str:
        """LangChain tool for vision text extraction"""
        return self.extract_text(image_path)
