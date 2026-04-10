"""Fluent builder for creating OCR output."""

from typing import Optional
from tests.harness.fakes.fake_ocr_service import OCROutput


class OCRResultBuilder:
    """Fluent builder for constructing OCR results.
    
    Example:
        >>> ocr = (OCRResultBuilder()
        ...     .with_text("GROCERY STORE\nMilk $3.99\nTotal $3.99")
        ...     .with_quality(0.85)
        ...     .using_easyocr()
        ...     .build())
    """
    
    def __init__(self):
        self._text = ""
        self._quality = 1.0
        self._method = "easyocr"
        self._confidence: Optional[float] = None
        self._processing_time_ms = 0.0
    
    def with_text(self, text: str) -> "OCRResultBuilder":
        """Set the extracted OCR text."""
        self._text = text
        return self
    
    def with_quality(self, score: float) -> "OCRResultBuilder":
        """Set quality score (0.0-1.0)."""
        self._quality = max(0.0, min(1.0, score))
        return self
    
    def with_high_quality(self) -> "OCRResultBuilder":
        """Set high quality score (0.9)."""
        self._quality = 0.9
        return self
    
    def with_medium_quality(self) -> "OCRResultBuilder":
        """Set medium quality score (0.6)."""
        self._quality = 0.6
        return self
    
    def with_low_quality(self) -> "OCRResultBuilder":
        """Set low quality score (0.2)."""
        self._quality = 0.2
        return self
    
    def using_easyocr(self) -> "OCRResultBuilder":
        """Set method to EasyOCR (local)."""
        self._method = "easyocr"
        return self
    
    def using_vision(self) -> "OCRResultBuilder":
        """Set method to Vision API (fallback)."""
        self._method = "vision"
        return self
    
    def using_fallback(self) -> "OCRResultBuilder":
        """Set method to fallback."""
        self._method = "fallback"
        return self
    
    def with_confidence(self, confidence: float) -> "OCRResultBuilder":
        """Set confidence level."""
        self._confidence = confidence
        return self
    
    def with_processing_time(self, ms: float) -> "OCRResultBuilder":
        """Set simulated processing time in milliseconds."""
        self._processing_time_ms = ms
        return self
    
    def build(self) -> OCROutput:
        """Build and return OCROutput."""
        return OCROutput(
            text=self._text,
            quality_score=self._quality,
            method=self._method,
            confidence=self._confidence,
            processing_time_ms=self._processing_time_ms
        )
    
    @classmethod
    def perfect_receipt(cls, merchant: str = "Store", total: str = "$10.00") -> "OCRResultBuilder":
        """Create high-quality OCR of a perfect receipt."""
        text = f"""{merchant}
Item 1 $5.00
Item 2 $5.00
Total {total}
Thank you!"""
        return cls().with_text(text).with_high_quality().using_easyocr()
    
    @classmethod
    def blurry_image(cls) -> "OCRResultBuilder":
        """Create low-quality OCR simulating blurry image."""
        return cls().with_text("ST... $..").with_low_quality().using_easyocr()
    
    @classmethod
    def empty_receipt(cls) -> "OCRResultBuilder":
        """Create OCR with no text (empty image)."""
        return cls().with_text("").with_quality(0.0)
