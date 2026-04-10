"""Fake OCR service for deterministic text extraction testing."""

from dataclasses import dataclass
from typing import Dict, Optional, Callable, Union
from pathlib import Path
import time

from .fake_component import FakeComponent, CallRecord, ConfigurationError


@dataclass
class OCROutput:
    """Structured output from OCR operation."""
    text: str
    quality_score: float = 1.0
    method: str = "easyocr"  # easyocr, vision, fallback
    confidence: Optional[float] = None
    processing_time_ms: float = 0.0


class FakeOCRService(FakeComponent):
    """Fake OCR service that returns configurable outputs.

    Simulates:
    - Local EasyOCR (default)
    - Vision API fallback
    - Quality scoring
    - Failures (OCRError, ImageProcessingError)
    - Sequential responses for retry testing

    Example:
        >>> ocr = FakeOCRService()
        >>> ocr.set_sequence([
        ...     ValueError("Network error"),
        ...     ValueError("Timeout"),
        ...     OCROutput(text="SUCCESS", quality=0.9)
        ... ])
        >>> text = ocr.extract_text("receipt.jpg")  # Raises
        >>> text = ocr.extract_text("receipt.jpg")  # Raises
        >>> text = ocr.extract_text("receipt.jpg")  # Returns "SUCCESS"
    """

    def __init__(self):
        super().__init__()
        self._image_outputs: Dict[str, OCROutput] = {}
        self._default_output: Optional[OCROutput] = None
        self._should_fail: bool = False
        self._failure_exception: Optional[Exception] = None
        self._attempt_count: int = 0
        self._max_attempts_before_success: int = 1
        self._sequence: list = []  # List of outputs/exceptions for sequential responses
        self._sequence_index: int = 0
        self._attempt_history: list = []  # Track each attempt with details

    def set_text_for_image(self, image_path: str, text: str, quality: float = 1.0) -> "FakeOCRService":
        """Configure OCR output for a specific image path.

        Args:
            image_path: Path to the image file
            text: OCR text to return
            quality: Quality score (0.0-1.0)

        Returns:
            Self for chaining
        """
        self._image_outputs[image_path] = OCROutput(
            text=text,
            quality_score=quality,
            method="easyocr"
        )
        return self

    def set_output_for_image(self, image_path: str, output: OCROutput) -> "FakeOCRService":
        """Set full OCROutput for an image."""
        self._image_outputs[image_path] = output
        return self

    def set_default_output(self, text: str, quality: float = 1.0, method: str = "easyocr") -> "FakeOCRService":
        """Set default output for images without specific config."""
        self._default_output = OCROutput(
            text=text,
            quality_score=quality,
            method=method
        )
        return self

    def set_quality_score(self, score: float) -> "FakeOCRService":
        """Set quality score for next OCR operation."""
        if self._default_output:
            self._default_output.quality_score = score
        return self

    def set_should_fail(
        self,
        exception: Exception,
        max_attempts: int = 1
    ) -> "FakeOCRService":
        """Configure OCR to fail with given exception.

        Args:
            exception: Exception to raise
            max_attempts: Number of attempts before succeeding (if 0, always fail)
        """
        self._should_fail = True
        self._failure_exception = exception
        self._max_attempts_before_success = max_attempts
        return self

    def set_fallback_output(
        self,
        image_path: str,
        text: str,
        quality: float = 1.0
    ) -> "FakeOCRService":
        """Configure fallback (Vision API) output for an image.

        Used when primary OCR quality is below threshold.
        """
        self._image_outputs[f"{image_path}__fallback"] = OCROutput(
            text=text,
            quality_score=quality,
            method="vision"
        )
        return self

    def set_sequence(self, sequence: list) -> "FakeOCRService":
        """Set a sequence of responses for sequential calls.

        Each element can be:
        - OCROutput for success
        - Exception for failure

        Example:
            >>> ocr.set_sequence([
            ...     ValueError("Network error"),
            ...     OCROutput(text="Success", quality=0.9)
            ... ])

        Args:
            sequence: List of outputs/exceptions to return in order

        Returns:
            Self for chaining
        """
        self._sequence = list(sequence)
        self._sequence_index = 0
        return self

    def get_attempt_history(self) -> list:
        """Get detailed history of all attempts.

        Returns list of dicts with:
        - attempt_number: int
        - success: bool
        - exception_type: str (if failed)
        - quality: float (if succeeded)
        """
        return list(self._attempt_history)

    def get_sequence_progress(self) -> tuple:
        """Get sequence progress as (current_index, total_length)."""
        return (self._sequence_index, len(self._sequence))

    def reset_attempts(self) -> None:
        """Reset attempt counter (useful for retry testing)."""
        self._attempt_count = 0

    def get_last_output(self) -> Optional[OCROutput]:
        """Get the output from the most recent call."""
        calls = self.get_calls("extract_text")
        if not calls:
            return None
        return calls[-1].result

    # ImageProcessingInterface implementation

    def preprocess_image(self, image_path: str) -> Dict[str, any]:
        """Fake image preprocessing - returns minimal metadata."""
        start = time.time()

        result = {
            "path": image_path,
            "preprocessed": True,
            "format": Path(image_path).suffix.lower()
        }

        duration_ms = (time.time() - start) * 1000
        self._record_call("preprocess_image", (image_path,), {}, result, duration_ms=duration_ms)
        return result

    def extract_text(self, image_path: str, use_vision_fallback: bool = False) -> str:
        """Extract text from image - returns configured output.

        Simulates failure behavior if configured.
        Uses sequence if configured, otherwise uses image-specific or default output.
        """
        start = time.time()
        self._attempt_count += 1
        attempt_number = self._attempt_count

        # Check if using sequence mode
        if self._sequence and self._sequence_index < len(self._sequence):
            item = self._sequence[self._sequence_index]
            self._sequence_index += 1

            # Track attempt history
            history_entry = {
                "attempt_number": attempt_number,
                "success": not isinstance(item, Exception),
                "exception_type": type(item).__name__ if isinstance(item, Exception) else None,
                "quality": item.quality_score if isinstance(item, OCROutput) else None,
            }
            self._attempt_history.append(history_entry)

            duration_ms = (time.time() - start) * 1000

            if isinstance(item, Exception):
                self._record_call(
                    "extract_text",
                    (image_path,),
                    {"use_vision_fallback": use_vision_fallback},
                    exception=item,
                    duration_ms=duration_ms
                )
                raise item
            else:
                # It's an OCROutput
                item.processing_time_ms = duration_ms
                self._record_call(
                    "extract_text",
                    (image_path,),
                    {"use_vision_fallback": use_vision_fallback},
                    result=item,
                    duration_ms=duration_ms
                )
                return item.text

        # Legacy mode: Check if should fail this attempt
        if self._should_fail and self._attempt_count <= self._max_attempts_before_success:
            duration_ms = (time.time() - start) * 1000

            # Track attempt history
            history_entry = {
                "attempt_number": attempt_number,
                "success": False,
                "exception_type": type(self._failure_exception).__name__,
                "quality": None,
            }
            self._attempt_history.append(history_entry)

            self._record_call(
                "extract_text",
                (image_path,),
                {"use_vision_fallback": use_vision_fallback},
                exception=self._failure_exception,
                duration_ms=duration_ms
            )
            raise self._failure_exception

        # Get configured output
        key = f"{image_path}__fallback" if use_vision_fallback else image_path
        output = self._image_outputs.get(key)

        if output is None:
            output = self._default_output

        if output is None:
            raise ConfigurationError(
                f"No OCR output configured for image: {image_path}. "
                f"Use set_text_for_image() or set_default_output()."
            )

        duration_ms = (time.time() - start) * 1000
        output.processing_time_ms = duration_ms

        # Track attempt history
        history_entry = {
            "attempt_number": attempt_number,
            "success": True,
            "exception_type": None,
            "quality": output.quality_score,
        }
        self._attempt_history.append(history_entry)

        self._record_call(
            "extract_text",
            (image_path,),
            {"use_vision_fallback": use_vision_fallback},
            result=output,
            duration_ms=duration_ms
        )

        return output.text

    def score_ocr_quality(self, text: str) -> float:
        """Score OCR quality - returns configured score or computed score."""
        start = time.time()

        # If we have a specific quality configured, use it
        last_call = self.get_calls("extract_text")
        if last_call and isinstance(last_call[-1].result, OCROutput):
            score = last_call[-1].result.quality_score
        else:
            # Simple heuristic for fake scoring
            score = min(1.0, len(text) / 100) if text else 0.0

        duration_ms = (time.time() - start) * 1000
        self._record_call("score_ocr_quality", (text,), {}, result=score, duration_ms=duration_ms)
        return score

    def get_attempt_count(self) -> int:
        """Get number of OCR attempts made."""
        return self._attempt_count

    def used_fallback(self) -> bool:
        """Check if fallback (Vision API) was used in last extraction."""
        calls = self.get_calls("extract_text")
        if not calls:
            return False
        output = calls[-1].result
        return isinstance(output, OCROutput) and output.method == "vision"
