"""Fake VisionProcessor for fast testing without EasyOCR"""
from typing import Dict, Any

from contracts.interfaces import ImageProcessingInterface


class FakeVisionProcessor(ImageProcessingInterface):
    """Fake image processor that returns pre-configured text without OCR"""

    def __init__(self, text: str = "MOCK RECEIPT TEXT", quality_score: float = 1.0):
        self.text = text
        self.quality_score = quality_score

    def preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """Fake preprocessing - returns empty dict"""
        return {}

    def extract_text(self, image_path: str, use_vision_fallback: bool = False) -> str:
        """Fake text extraction - returns configured text"""
        return self.text

    def score_ocr_quality(self, text: str) -> float:
        """Fake quality scoring - returns configured score"""
        return self.quality_score
