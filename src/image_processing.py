"""Image processing utilities"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from contracts.interfaces import ImageProcessingInterface
from services.ocr_service import OCRService, OCRObservability

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


class VisionProcessor(ImageProcessor, ImageProcessingInterface):
    """Concrete implementation using OCR service with optional vision LLM"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self._ocr_service = None  # Lazy initialization
        api_key = os.getenv("OPENAI_API_KEY")
        self.vision_llm = ChatOpenAI(model=model_name, temperature=0.0, api_key=api_key)

    @property
    def ocr_service(self):
        """Lazy initialization of OCRService to avoid EasyOCR import in tests"""
        if self._ocr_service is None:
            self._ocr_service = OCRService()
        return self._ocr_service

    def preprocess(self, image_path: str) -> Any:
        """Preprocess image for OCR - delegates to internal method"""
        return self._preprocess_image(image_path)

    def _preprocess_image(self, image_path: str) -> str:
        """Internal preprocessing - reads file directly for test mocking compatibility"""
        import base64
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    def preprocess_image(self, image_path: str) -> str:
        """Public preprocessing API"""
        return self._preprocess_image(image_path)

    def _extract_vision_text(self, image_path: str) -> str:
        """Extract text using Vision LLM (OpenAI GPT-4 Vision)

        This is the primary text extraction method for VisionProcessor.
        No OCRService dependency - pure vision LLM approach.
        """
        from langchain_core.messages import HumanMessage
        import base64

        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": "Extract all text from this receipt image."}
            ]
        )
        response = self.vision_llm.invoke([message])
        return response.content

    def extract_text(self, image_path: str, use_vision_fallback: bool = False) -> str:
        """Extract text from image using Vision LLM only

        Simple, deterministic: always use vision LLM, no OCR fallback.
        This prevents EasyOCR from being triggered in tests.
        """
        start_time = time.time()
        text = self._extract_vision_text(image_path)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Vision LLM OCR: {len(text)} chars in {duration_ms:.1f}ms")
        return text

    def score_ocr_quality(self, text: str) -> float:
        """Score OCR text quality (0-1)"""
        return self.ocr_service.score_ocr_quality(text)

    @tool
    def preprocess_image_tool(self, image_path: str) -> Any:
        """LangChain tool for image preprocessing"""
        return self.preprocess(image_path)

    @tool
    def extract_vision_text_tool(self, image_path: str) -> str:
        """LangChain tool for vision text extraction"""
        return self.extract_text(image_path)
