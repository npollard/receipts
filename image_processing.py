"""Image processing utilities"""

import logging
from abc import ABC, abstractmethod
from typing import Any
import cv2
import pytesseract
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

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


class OCRProcessor(ImageProcessor):
    """Concrete implementation of image processing using OpenCV and Tesseract"""

    def __init__(self, threshold: int = 150):
        self.threshold = threshold
        self._preprocess_chain = RunnableLambda(self._preprocess_image)
        self._ocr_chain = RunnableLambda(self._extract_ocr_text)
        logger.info(f"Initialized OCRProcessor with threshold: {threshold}")

    def preprocess(self, image_path: str) -> Any:
        """Public interface for preprocessing"""
        return self._preprocess_image(image_path)

    def extract_text(self, image_path: str) -> str:
        """Public interface for OCR extraction"""
        return self._extract_ocr_text(image_path)

    def _preprocess_image(self, image_path: str) -> Any:
        """Internal preprocessing logic"""
        logger.debug(f"Preprocessing image: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)[1]
        return img

    def _extract_ocr_text(self, image_path: str) -> str:
        """Internal OCR extraction logic"""
        logger.debug(f"Extracting OCR text from: {image_path}")
        processed_img = self._preprocess_image(image_path)
        text = pytesseract.image_to_string(processed_img)
        logger.info(f"Extracted {len(text)} characters from {image_path}")
        logger.info(f"Extracted text: {text[:100]}...")  # Log first 100 chars
        return text

    @tool
    def preprocess_image_tool(self, image_path: str) -> Any:
        """LangChain tool for image preprocessing"""
        return self.preprocess(image_path)

    @tool
    def extract_ocr_text_tool(self, image_path: str) -> str:
        """LangChain tool for OCR extraction"""
        return self.extract_text(image_path)
