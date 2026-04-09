"""OCR service for extracting text from images using local EasyOCR

This service enforces thread limits to prevent hidden parallelism.
All OCR operations respect the centralized RuntimeConfig.
"""

import re
import os
import sys
import time
import warnings
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain_core.runnables import RunnableLambda

# Thread limits must be set BEFORE importing torch/easyocr
# The main entrypoint should have already enforced these
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Detect MPS device and suppress torch warnings before importing EasyOCR
try:
    import torch
    # Enforce thread limits on torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.set_num_interop_threads(1)
    is_mps = torch.backends.mps.is_available()
    if is_mps:
        # Suppress pin_memory warning on MPS devices
        warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)
except ImportError:
    is_mps = False

import easyocr
from PIL import Image
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from core.file_operations import encode_file_base64
from core.logging import get_ocr_logger
from core.exceptions import (
    OCRError, ImageProcessingError, TextExtractionError,
    VisionAPIError, QualityScoreError
)
from config.ocr_config import OCRConfig, ENV_OCR_CONFIG
from contracts.interfaces import ImageProcessingInterface

logger = get_ocr_logger(__name__)


@dataclass
class OCRObservability:
    """Observability data for OCR operations"""
    method: str  # 'easyocr', 'vision', 'fallback'
    start_time: float
    end_time: float
    quality_score: float
    text_length: int
    confidence_threshold: float

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'duration_ms': round(self.duration_ms, 2),
            'quality_score': round(self.quality_score, 3),
            'text_length': self.text_length,
            'confidence_threshold': self.confidence_threshold,
        }


class OCRService(ImageProcessingInterface):
    """Service for OCR text extraction using local EasyOCR"""

    def __init__(self, config: OCRConfig = None, use_gpu: bool = None, lang: List[str] = None,
                 confidence_threshold: float = None, debug: bool = None, comparison_mode: bool = None,
                 quality_threshold: float = None, debug_ocr: bool = None):
        """
        Initialize EasyOCR service with configurable settings and thread control

        Args:
            config: OCRConfig instance (takes precedence over individual parameters)
            use_gpu: Whether to use GPU acceleration (overrides config if provided)
            lang: List of language codes for OCR (overrides config if provided)
            confidence_threshold: Minimum confidence score for text extraction (overrides config if provided)
            debug: Enable debug logging for OCR pipeline stages (overrides config if provided)
            comparison_mode: Enable comparison with OpenAI Vision OCR for evaluation (overrides config if provided)
            quality_threshold: Minimum OCR quality score before fallback to Vision OCR (overrides config if provided)
            debug_ocr: Enable detailed OCR decision observability (overrides config if provided)
        """
        # Use provided config or environment-based config
        if config is None:
            config = ENV_OCR_CONFIG

        # Override config with explicit parameters if provided
        if use_gpu is not None:
            config = config.override(use_gpu=use_gpu)
        if lang is not None:
            config = config.override(languages=lang)
        if confidence_threshold is not None:
            config = config.override(confidence_threshold=confidence_threshold)
        if debug is not None:
            config = config.override(debug=debug)
        if comparison_mode is not None:
            config = config.override(comparison_mode=comparison_mode)
        if quality_threshold is not None:
            config = config.override(quality_threshold=quality_threshold)
        if debug_ocr is not None:
            config = config.override(debug_ocr=debug_ocr)

        # Store configuration
        self.config = config

        # Set instance attributes for backward compatibility
        self.use_gpu = config.use_gpu
        self.lang = config.languages
        self.confidence_threshold = config.confidence_threshold
        self.debug = config.debug
        self.comparison_mode = config.comparison_mode
        self.quality_threshold = config.quality_threshold
        self.debug_ocr = config.debug_ocr

        # Log thread limits being used
        torch_threads = torch.get_num_threads() if 'torch' in sys.modules else 'N/A'
        logger.info(f"Initializing OCRService with thread limits: torch={torch_threads}, "
                   f"OMP={os.environ.get('OMP_NUM_THREADS', 'N/A')}")

        # Detect test environment and disable EasyOCR to prevent model downloads
        self._is_test_mode = "PYTEST_CURRENT_TEST" in os.environ
        if self._is_test_mode:
            logger.info("Test environment detected - EasyOCR disabled")
            self._ocr_reader = None
        else:
            # Initialize EasyOCR reader lazily for production
            self._ocr_reader = None

        # Initialize OpenAI Vision client for fallback and comparison mode
        self.vision_llm = ChatOpenAI(
            model=config.openai_model,
            temperature=config.openai_temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )

        self._preprocess_chain = RunnableLambda(self._preprocess_image)
        self._ocr_chain = RunnableLambda(self._extract_easyocr_text)

        logger.info(f"Initialized OCRService with EasyOCR (GPU: {config.use_gpu}, Lang: {config.languages}, "
                   f"Confidence: {config.confidence_threshold}, Quality Threshold: {config.quality_threshold})")

    def _get_reader(self):
        """Lazy initialization of EasyOCR reader

        This prevents EasyOCR from downloading models or initializing
        during tests when OCRService is instantiated but not used.
        """
        if self._is_test_mode:
            raise RuntimeError("EasyOCR is disabled in test environment. Use mocks or fake implementations.")
        if self._ocr_reader is None:
            self._ocr_reader = easyocr.Reader(self.config.languages, gpu=self.config.use_gpu)
        return self._ocr_reader

    def preprocess_image(self, image_path: str) -> Any:
        """Preprocess image for OCR"""
        return self._preprocess_image(image_path)

    def extract_text_with_observability(self, image_path: str, use_vision_fallback: bool = False) -> Tuple[str, OCRObservability]:
        """Extract text with full observability data - internal use

        Args:
            image_path: Path to the image file
            use_vision_fallback: If True, use Vision OCR directly instead of EasyOCR

        Returns:
            Tuple of (extracted_text, observability_data)
        """
        start_time = time.time()

        if use_vision_fallback:
            logger.info("Using Vision OCR fallback")
            text = self._extract_vision_text(image_path)
            end_time = time.time()
            obs = OCRObservability(
                method='vision',
                start_time=start_time,
                end_time=end_time,
                quality_score=0.0,  # Not scored for direct vision
                text_length=len(text),
                confidence_threshold=self.confidence_threshold,
            )
            return text, obs

        if self.comparison_mode:
            text = self._extract_with_comparison(image_path)
        else:
            text, method, quality = self._extract_with_quality_fallback_observable(image_path)
            end_time = time.time()
            obs = OCRObservability(
                method=method,
                start_time=start_time,
                end_time=end_time,
                quality_score=quality,
                text_length=len(text),
                confidence_threshold=self.confidence_threshold,
            )
            return text, obs

        # Comparison mode doesn't return observability
        end_time = time.time()
        obs = OCRObservability(
            method='comparison',
            start_time=start_time,
            end_time=end_time,
            quality_score=0.0,
            text_length=len(text),
            confidence_threshold=self.confidence_threshold,
        )
        return text, obs

    def extract_text(self, image_path: str, use_vision_fallback: bool = False) -> str:
        """Extract text from image - implements ImageProcessingInterface

        Args:
            image_path: Path to the image file
            use_vision_fallback: If True, use Vision OCR directly instead of EasyOCR

        Returns:
            Extracted text string only (for interface compatibility)
        """
        text, _ = self.extract_text_with_observability(image_path, use_vision_fallback)
        return text

    def _log_debug_header(self, image_path: str):
        """Log debug header for OCR processing"""
        logger.debug("="*60)
        logger.debug(f"OCR DECISION DEBUG FOR: {os.path.basename(image_path)}")
        logger.debug("="*60)
        logger.debug(f"Quality Threshold: {self.quality_threshold:.3f}")
        logger.debug("Debug Mode: ENABLED")

    def _log_ocr_results(self, text: str, quality_score: float, ocr_type: str = "EASYOCR"):
        """Log OCR results with quality information"""
        logger.debug(f"--- {ocr_type} RESULTS ---")
        logger.debug(f"Quality Score: {quality_score:.3f}")
        logger.debug(f"Text Length: {len(text)} characters")
        logger.debug(f"Text Preview: {text[:200]}{'...' if len(text) > 200 else ''}")

    def _log_quality_breakdown(self, reasoning: dict, ocr_type: str = "EASYOCR"):
        """Log detailed quality score breakdown"""
        logger.debug(f"--- {ocr_type} QUALITY BREAKDOWN ---")
        scores = reasoning['component_scores']
        logger.debug(f"Text Length: {scores['text_length']:.1f}/20")
        logger.debug(f"Price Patterns: {scores['price_patterns']:.1f}/25")
        logger.debug(f"Total Keywords: {scores['total_keyword']:.1f}/20")
        logger.debug(f"Word Quality: {scores['word_quality']:.1f}/25")
        logger.debug(f"Noise Penalty: -{scores['noise_penalty']:.1f}/10")
        total_score = sum(scores.values()) - 2*scores['noise_penalty']
        logger.debug(f"Total Raw Score: {total_score:.1f}/100")

    def _log_fallback_decision(self, quality_score: float, should_fallback: bool, reasoning: dict):
        """Log fallback decision with reasoning"""
        logger.debug("--- FALLBACK DECISION ---")
        logger.debug(f"Quality Score ({quality_score:.3f}) < Threshold ({self.quality_threshold:.3f}): {should_fallback}")
        logger.debug(f"Fallback Required: {'YES' if should_fallback else 'NO'}")

        if should_fallback:
            logger.debug(f"Reasoning: {reasoning['reasoning']}")
        else:
            logger.debug("Reasoning: Good quality OCR - using EasyOCR result")

    def _log_final_decision(self, selected_ocr: str, reason: str):
        """Log final OCR selection decision"""
        logger.debug("--- FINAL DECISION ---")
        logger.debug(f"Selected OCR: {selected_ocr}")
        logger.debug(f"Reason: {reason}")

    def _log_final_output(self, text: str):
        """Log final OCR output"""
        logger.debug("--- FINAL OUTPUT ---")
        logger.debug(f"Selected Text Length: {len(text)} characters")
        logger.debug(f"Selected Text Preview: {text[:300]}{'...' if len(text) > 300 else ''}")
        logger.debug("="*60)

    def _log_error_handling(self, error: Exception):
        """Log error handling information"""
        logger.debug("--- ERROR HANDLING ---")
        logger.debug(f"Error: {str(error)}")
        logger.debug("Initiating emergency fallback to basic EasyOCR")

    def _log_catastrophic_failure(self, primary_error: Exception, fallback_error: Exception):
        """Log catastrophic failure information"""
        logger.debug("--- CATASTROPHIC FAILURE ---")
        logger.debug("Both EasyOCR and fallback failed")
        logger.debug(f"Primary Error: {str(primary_error)}")
        logger.debug(f"Fallback Error: {str(fallback_error)}")
        logger.debug("="*60)

    def _extract_with_quality_fallback(self, image_path: str) -> str:
        """Extract text using EasyOCR with automatic fallback to Vision OCR based on quality score"""
        text, _, _ = self._extract_with_quality_fallback_observable(image_path)
        return text

    def _extract_with_quality_fallback_observable(self, image_path: str) -> Tuple[str, str, float]:
        """Extract text with observability - returns (text, method_used, quality_score)"""
        try:
            if self.debug_ocr:
                self._log_debug_header(image_path)

            # Extract text using EasyOCR
            easyocr_text = self._extract_easyocr_text(image_path)
            quality_score = self.score_ocr_quality(easyocr_text)

            logger.info(f"EasyOCR Quality Score: {quality_score:.3f} (threshold: {self.quality_threshold:.3f}) for {image_path}")

            if self.debug_ocr:
                reasoning = self.get_fallback_reasoning(easyocr_text, self.quality_threshold)
                self._log_ocr_results(easyocr_text, quality_score, "EASYOCR")
                self._log_quality_breakdown(reasoning, "EASYOCR")

            # Determine if fallback is needed
            should_fallback = self.should_fallback(easyocr_text, self.quality_threshold)

            if self.debug_ocr:
                self._log_fallback_decision(quality_score, should_fallback, reasoning)

            if should_fallback:
                return self._handle_fallback_to_vision(image_path, easyocr_text, quality_score)
            else:
                return self._use_easyocr_result(easyocr_text, quality_score)

        except Exception as e:
            return self._handle_extraction_error(e, image_path)

    def _handle_fallback_to_vision(self, image_path: str, easyocr_text: str, easyocr_quality: float) -> Tuple[str, str, float]:
        """Handle fallback to Vision OCR when EasyOCR quality is insufficient

        Returns: (text, method, quality_score)
        """
        logger.warning(f"EasyOCR quality below threshold - falling back to Vision OCR for {image_path}")

        if self.debug_ocr:
            logger.debug("--- INITIATING FALLBACK TO VISION OCR ---")

        # Extract text using Vision OCR
        vision_text = self._extract_vision_text(image_path)
        vision_quality_score = self.score_ocr_quality(vision_text)

        logger.info(f"Vision OCR Quality Score: {vision_quality_score:.3f} for {image_path}")

        if self.debug_ocr:
            vision_reasoning = self.get_fallback_reasoning(vision_text, self.quality_threshold)
            self._log_ocr_results(vision_text, vision_quality_score, "VISION OCR")
            self._log_quality_breakdown(vision_reasoning, "VISION OCR")
            self._log_final_decision("VISION OCR", "EasyOCR quality below threshold")

        return vision_text, 'fallback', vision_quality_score

    def _use_easyocr_result(self, easyocr_text: str, quality_score: float) -> Tuple[str, str, float]:
        """Use EasyOCR result when quality is acceptable

        Returns: (text, method, quality_score)
        """
        logger.info("Using EasyOCR result")

        if self.debug_ocr:
            self._log_final_decision("EASYOCR", "Quality acceptable")
            self._log_final_output(easyocr_text)

        return easyocr_text, 'easyocr', quality_score

    def _handle_extraction_error(self, error: Exception, image_path: str) -> Tuple[str, str, float]:
        """Handle extraction errors with fallback mechanisms

        Returns: (text, method, quality_score)
        """
        logger.error(f"Error in quality-based OCR extraction for {image_path}: {str(error)}")

        if self.debug_ocr:
            self._log_error_handling(error)

        # Check if this is the EasyOCR library bug
        if isinstance(error, TextExtractionError) and "EasyOCR library bug" in str(error):
            logger.warning(f"EasyOCR library bug detected, attempting Vision OCR fallback for {image_path}")
            try:
                vision_text = self._extract_vision_text(image_path)
                vision_quality = self.score_ocr_quality(vision_text)
                return vision_text, 'fallback', vision_quality
            except Exception as vision_error:
                logger.error(f"Vision OCR fallback also failed for {image_path}: {str(vision_error)}")
                if self.debug_ocr:
                    self._log_catastrophic_failure(error, vision_error)
                raise TextExtractionError(f"All OCR methods failed for {image_path}. EasyOCR bug: {str(error)}, Vision error: {str(vision_error)}")
        else:
            # Standard fallback to basic EasyOCR without quality scoring
            try:
                easyocr_text = self._extract_easyocr_text(image_path)
                easyocr_quality = self.score_ocr_quality(easyocr_text)
                return easyocr_text, 'easyocr', easyocr_quality
            except Exception as fallback_error:
                logger.error(f"Fallback EasyOCR also failed for {image_path}: {str(fallback_error)}")

                if self.debug_ocr:
                    self._log_catastrophic_failure(error, fallback_error)

                # Final fallback to Vision OCR
                try:
                    logger.warning(f"Attempting final Vision OCR fallback for {image_path}")
                    vision_text = self._extract_vision_text(image_path)
                    vision_quality = self.score_ocr_quality(vision_text)
                    return vision_text, 'fallback', vision_quality
                except Exception as vision_error:
                    logger.error(f"Final Vision OCR fallback also failed for {image_path}: {str(vision_error)}")
                    raise TextExtractionError(f"All OCR methods failed for {image_path}. Primary error: {str(error)}, EasyOCR fallback: {str(fallback_error)}, Vision fallback: {str(vision_error)}")

    def _extract_with_comparison(self, image_path: str) -> str:
        """Extract text using both EasyOCR and OpenAI Vision for comparison"""
        try:
            # Get EasyOCR result
            easyocr_result = self._extract_easyocr_text(image_path)

            # Get OpenAI Vision result
            vision_result = self._extract_vision_text(image_path)

            # Log comparison results
            self._log_comparison(image_path, easyocr_result, vision_result)

            # Return EasyOCR result for normal operation
            return easyocr_result

        except Exception as e:
            logger.error(f"Comparison mode error for {image_path}: {str(e)}")
            # Fallback to EasyOCR only
            return self._extract_easyocr_text(image_path)

    def _extract_vision_text(self, image_path: str) -> str:
        """Extract text using OpenAI Vision API"""
        try:
            # Convert image to base64
            base64_image = encode_file_base64(image_path)

            vision_prompt = """
            Extract all text from this receipt image. Return ONLY the extracted text
            without any additional commentary, formatting, or explanations.
            Preserve the layout and structure as much as possible.
            """

            message = HumanMessage(
                content=[
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto"
                        }
                    }
                ]
            )

            response = self.vision_llm.invoke([message])
            extracted_text = response.content.strip()

            return extracted_text

        except Exception as e:
            logger.error(f"OpenAI Vision API error for {image_path}: {str(e)}")
            return f"Vision OCR Error: {str(e)}"

    def _log_comparison(self, image_path: str, easyocr_text: str, vision_text: str):
        """Log comparison between EasyOCR and OpenAI Vision results"""
        logger.info("="*80)
        logger.info(f"OCR COMPARISON: {image_path}")
        logger.info("="*80)

        logger.info("-"*60)
        logger.info("EASYOCR RESULT:")
        logger.info("-"*60)
        logger.info(f"Length: {len(easyocr_text)} characters")
        logger.info(f"Text:\n{easyocr_text}")

        logger.info("-"*60)
        logger.info("OPENAI VISION RESULT:")
        logger.info("-"*60)
        logger.info(f"Length: {len(vision_text)} characters")
        logger.info(f"Text:\n{vision_text}")

        logger.info("-"*60)
        logger.info("COMPARISON SUMMARY:")
        logger.info("-"*60)
        logger.info(f"EasyOCR length:  {len(easyocr_text)} chars")
        logger.info(f"Vision length:  {len(vision_text)} chars")
        logger.info(f"Length diff:    {len(easyocr_text) - len(vision_text)} chars")

        # Calculate word overlap
        easyocr_words = set(easyocr_text.lower().split())
        vision_words = set(vision_text.lower().split())
        common_words = easyocr_words.intersection(vision_words)

        logger.info(f"EasyOCR words:  {len(easyocr_words)}")
        logger.info(f"Vision words:   {len(vision_words)}")
        logger.info(f"Common words:   {len(common_words)}")
        logger.info(f"Word overlap:   {len(common_words) / max(len(easyocr_words), len(vision_words)) * 100:.1f}%")

        logger.info("="*80)

    def _preprocess_image(self, image_path: str) -> dict:
        """Preprocess image for EasyOCR"""
        try:
            if not os.path.exists(image_path):
                raise ImageProcessingError(f"Image file not found: {image_path}")

            # Load and validate image
            try:
                image = Image.open(image_path)
                image.verify()  # Verify image integrity

                # Re-open image after verify (PIL requirement)
                image = Image.open(image_path)
            except (IOError, OSError) as e:
                raise ImageProcessingError(f"Invalid or corrupted image file: {image_path} - {str(e)}")

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                try:
                    image = image.convert('RGB')
                except Exception as e:
                    raise ImageProcessingError(f"Failed to convert image to RGB: {image_path} - {str(e)}")

            # Convert to numpy array for EasyOCR
            try:
                image_array = np.array(image)

                # Defensive validation of image array
                if image_array is None or not isinstance(image_array, np.ndarray):
                    raise ImageProcessingError("Invalid image array")

                if image_array.size == 0:
                    raise ImageProcessingError("Empty image array")

                # Debug logging after preprocessing
                logger.debug(f"Preprocessed image shape: {image_array.shape}, dtype: {image_array.dtype}")

            except Exception as e:
                raise ImageProcessingError(f"Failed to convert image to array: {image_path} - {str(e)}")

            return {
                'image_path': image_path,
                'image_array': image_array,
                'original_size': image.size,
                'mode': image.mode
            }

        except ImageProcessingError:
            # Re-raise specific exceptions
            raise
        except Exception as e:
            # Catch any unexpected errors and wrap them
            raise ImageProcessingError(f"Unexpected error preprocessing image {image_path}: {str(e)}")

    def _extract_easyocr_text(self, image_input: Any) -> str:
        """Extract text using EasyOCR"""
        if isinstance(image_input, str):
            # If it's a string path, preprocess first
            processed = self._preprocess_image(image_input)
        else:
            processed = image_input

        # Validate processed data before OCR
        if not isinstance(processed, dict):
            raise TextExtractionError("Invalid processed data format: expected dictionary")

        if 'image_array' not in processed:
            raise TextExtractionError("Missing 'image_array' in processed data")

        if not isinstance(processed['image_array'], np.ndarray):
            raise TextExtractionError("Invalid image_array type: expected numpy array")

        if processed['image_array'].size == 0:
            raise TextExtractionError("Empty image array provided")

        # Early failure guard before EasyOCR call
        if processed['image_array'] is None or processed['image_array'].size == 0:
            raise TextExtractionError("Invalid preprocessed image")

        try:
            # Run EasyOCR (lazy initialization)
            results = self._get_reader().readtext(processed['image_array'])

            # Debug: Log raw OCR output
            if self.debug:
                logger.debug(f"RAW OCR OUTPUT for {processed['image_path']}:")
                for i, (bbox, text, confidence) in enumerate(results):
                    logger.debug(f"  {i+1:2d}. [{confidence:.3f}] {text[:50]}{'...' if len(text) > 50 else ''}")

            # Extract and combine text from results
            text_lines = []
            confidence_scores = []

            if results:  # Check if OCR found any text
                for (bbox, text, confidence) in results:
                    if text and confidence >= self.confidence_threshold:  # Filter by configurable threshold
                        text_lines.append(text.strip())
                        confidence_scores.append(confidence)

            # Combine text with proper spacing and layout preservation
            extracted_text = self._combine_text_lines(text_lines, results)

            # Debug: Log filtered output
            if self.debug:
                logger.debug(f"FILTERED OUTPUT for {processed['image_path']} (threshold: {self.confidence_threshold}):")
                for i, line in enumerate(text_lines):
                    logger.debug(f"  {i+1:2d}. {line[:50]}{'...' if len(line) > 50 else ''}")
                logger.debug(f"Combined text length: {len(extracted_text)} characters")

            # Apply text normalization
            normalized_text = self._normalize_text(extracted_text)

            # Debug: Log final normalized text
            if self.debug:
                logger.debug(f"FINAL NORMALIZED TEXT for {processed['image_path']}:")
                logger.debug(f"  Length: {len(normalized_text)} characters")
                logger.debug(f"  Preview: {normalized_text[:200]}{'...' if len(normalized_text) > 200 else ''}")

            # Log extraction results
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            logger.info(f"Extracted {len(normalized_text)} characters from {processed['image_path']} "
                       f"(avg confidence: {avg_confidence:.2f})")

            return normalized_text

        except Exception as e:
            # Handle any OCR processing errors with proper context
            image_path = processed.get('image_path', 'unknown')
            logger.exception(f"OCR extraction failed for {image_path}")
            raise TextExtractionError(f"OCR processing failed for {image_path}: {str(e)}") from e

    def _combine_text_lines(self, text_lines: List[str], ocr_results: List) -> str:
        """Combine OCR text lines preserving layout structure with line reconstruction"""
        if not text_lines or not ocr_results:
            return ""

        # Create list of (y_coord, x_coord, text) for filtered results
        filtered_items = []

        # Map text_lines to their corresponding OCR results
        text_index = 0
        for bbox, text, confidence in ocr_results:
            # Only include results that passed the confidence filter
            if text and confidence >= self.confidence_threshold and text_index < len(text_lines):
                # Use top-left corner coordinates
                y_coord = bbox[0][1] if bbox else 0
                x_coord = bbox[0][0] if bbox else 0

                filtered_items.append({
                    'y_coord': y_coord,
                    'x_coord': x_coord,
                    'text': text_lines[text_index],
                    'bbox': bbox
                })
                text_index += 1

    # Group items into lines based on Y-coordinate proximity
        lines = self._group_into_lines(filtered_items)

        # Sort lines by Y-coordinate (top-to-bottom)
        lines.sort(key=lambda line: line['y_coord'])

        # Extract text from each line
        reconstructed_lines = []
        for line in lines:
            # Sort items within line by X-coordinate (left-to-right)
            line['items'].sort(key=lambda item: item['x_coord'])

            # Combine items on the same line with spaces
            line_text = ' '.join(item['text'] for item in line['items'])
            reconstructed_lines.append(line_text)

        # Join lines with newlines to create receipt-like structure
        combined_text = '\n'.join(reconstructed_lines)

        return combined_text

    def _group_into_lines(self, items: List[Dict]) -> List[Dict]:
        """Group OCR items into lines based on Y-coordinate proximity"""
        if not items:
            return []

        # Calculate line height threshold (average height * 0.5)
        heights = []
        for item in items:
            bbox = item['bbox']
            if bbox and len(bbox) >= 2:
                height = abs(bbox[0][1] - bbox[2][1])  # Difference between top-left and bottom-left Y
                heights.append(height)

        avg_height = np.mean(heights) if heights else 20
        line_threshold = avg_height * 0.5  # Items within 50% of average height are on same line

        # Group items into lines
        lines = []
        current_line = None

        for item in sorted(items, key=lambda x: x['y_coord']):
            if current_line is None:
                # Start first line
                current_line = {
                    'y_coord': item['y_coord'],
                    'items': [item]
                }
            else:
                # Check if item belongs to current line
                y_diff = abs(item['y_coord'] - current_line['y_coord'])
                if y_diff <= line_threshold:
                    # Add to current line
                    current_line['items'].append(item)
                    # Update line Y-coordinate to average of items
                    current_line['y_coord'] = np.mean([i['y_coord'] for i in current_line['items']])
                else:
                    # Start new line
                    lines.append(current_line)
                    current_line = {
                        'y_coord': item['y_coord'],
                        'items': [item]
                    }

        # Add the last line
        if current_line:
            lines.append(current_line)

        return lines

    def _normalize_text(self, text: str) -> str:
        """Apply conservative text normalization that preserves readability"""
        if not text:
            return ""

        # Remove excessive whitespace but preserve line breaks for structure
        text = re.sub(r'[ \t]+', ' ', text)  # Only normalize spaces and tabs
        text = re.sub(r'\n\s*\n', '\n', text)  # Remove excessive blank lines

        # Apply conservative character corrections (only obvious OCR errors)
        text = self._fix_character_errors(text)

        # Apply minimal receipt-specific fixes (only clear OCR errors)
        text = self._fix_receipt_patterns(text)

        return text.strip()

    def _fix_character_errors(self, text: str) -> str:
        """Fix common OCR character errors using context-aware heuristics"""
        # Process text word by word for context-aware corrections
        words = text.split()
        corrected_words = []

        for word in words:
            corrected_word = self._fix_word_character_errors(word)
            corrected_words.append(corrected_word)

        return ' '.join(corrected_words)

    def _fix_word_character_errors(self, word: str) -> str:
        """Fix character errors within a word using context heuristics"""
        if not word:
            return word

        # Extract prefix/suffix punctuation
        prefix_punct = ''
        suffix_punct = ''
        clean_word = word

        # Handle common punctuation patterns
        if word and word[0] in '.,;:!?()[]{}"\'':
            prefix_punct = word[0]
            clean_word = word[1:]
        if clean_word and clean_word[-1] in '.,;:!?()[]{}"\'':
            suffix_punct = clean_word[-1]
            clean_word = clean_word[:-1]

        # Apply generalized character corrections based on context
        corrected_clean_word = self._apply_context_corrections(clean_word)

        # Reconstruct word with original punctuation
        return prefix_punct + corrected_clean_word + suffix_punct

    def _apply_context_corrections(self, word: str) -> str:
        """Apply context-aware character corrections to a clean word"""
        if not word:
            return word

        corrected = word

        # Apply generalized character corrections without hardcoded mappings
        corrected = self._apply_generalized_corrections(word)

        return corrected

    def _apply_generalized_corrections(self, word: str) -> str:
        """Apply generalized character corrections using pattern matching"""
        if not word:
            return word

        corrected = word

        # Fix common character confusions
        corrected = self._fix_character_confusions(corrected)

        # Fix price formatting issues
        corrected = self._fix_price_formatting(corrected)

        return corrected

    def _fix_character_confusions(self, word: str) -> str:
        """Fix common OCR character confusions using conservative pattern matching"""
        if not word:
            return word

        corrected = word

        # Only apply substitutions in clearly numeric contexts
        # Pattern 1: Word is primarily numeric (e.g., "12345", "$12.99")
        if re.match(r'^[\$]?\d+[\.,]?\d*$', word):
            # Apply conservative substitutions only for obvious OCR errors
            substitutions = {
                'O': '0',  # Letter O to number 0
                'I': '1',  # Letter I to number 1
                'S': '5',  # Letter S to number 5 - only in pure numeric context
            }
            for old_char, new_char in substitutions.items():
                corrected = corrected.replace(old_char, new_char)

        # Pattern 2: Word contains clear price/amount patterns
        elif re.search(r'\$\d+[\.,]\d{2}|\d+[\.,]\d{2}\$', word):
            # Apply substitutions only to numeric parts
            numeric_part = re.sub(r'[^\d.,$]', '', word)
            if numeric_part:
                substitutions = {
                    'O': '0',
                    'I': '1',
                    'S': '5',
                }
                for old_char, new_char in substitutions.items():
                    corrected = corrected.replace(old_char, new_char)

        return corrected

    def _fix_price_formatting(self, word: str) -> str:
        """Fix common price formatting issues in OCR text"""
        if not word:
            return word

        corrected = word

        # Fix space-separated decimals (e.g., "6 . 49" -> "6.49")
        if re.search(r'\d+\s+\.\s+\d{2}', corrected):
            corrected = re.sub(r'(\d+)\s+\.\s+(\d{2})', r'\1.\2', corrected)

        # Fix comma decimals (e.g., "6,49" -> "6.49")
        if re.search(r'\d+,\d{2}', corrected):
            corrected = corrected.replace(',', '.')

        return corrected

    def _fix_receipt_patterns(self, text: str) -> str:
        """Fix common receipt-specific patterns and formatting issues"""
        if not text:
            return text

        corrected = text

        # Fix common OCR errors in receipt text
        corrections = [
            # Common character substitutions
            (r'\bT0TAL\b', 'TOTAL'),
            (r'\bAM0UNT\b', 'AMOUNT'),
            (r'\bSUBT0TAL\b', 'SUBTOTAL'),
            (r'\bCASHIER\b', 'CASHIER'),
            (r'\bCUST0MER\b', 'CUSTOMER'),
        ]

        for pattern, replacement in corrections:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

        return corrected

    def get_ocr_confidence(self, text: str) -> float:
        """Calculate overall OCR confidence score"""
        if not text:
            return 0.0

        # Use the quality scoring system to determine confidence
        quality_score = self.score_ocr_quality(text)

        # Convert quality score (0-1) to confidence percentage (0-100)
        confidence = quality_score * 100.0

        return confidence

    def extract_text_with_confidence(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image with confidence scoring"""
        try:
            # Extract text using EasyOCR
            extracted_text = self.extract_text(image_path)

            # Calculate confidence score
            confidence = self.get_ocr_confidence(extracted_text)

            return {
                'text': extracted_text,
                'confidence': confidence,
                'method': 'easyocr',
                'success': True
            }

        except Exception as e:
            logger.error(f"Error extracting text with confidence: {str(e)}")
            return {
                'text': '',
                'confidence': 0.0,
                'method': 'easyocr',
                'success': False,
                'error': str(e)
            }

    def extract_receipt_fields(self, text: str) -> Dict[str, Any]:
        """Extract structured receipt fields from OCR text"""
        if not text:
            return {
                'total_amount': None,
                'items': [],
                'merchant_name': None,
                'date': None,
                'confidence': 0.0
            }

        # Extract various fields
        total_amount = self._extract_total_amount(text)
        items = self._extract_items(text)
        merchant_name = self._extract_merchant_name(text)
        date = self._extract_date(text)

        # Calculate overall confidence
        confidence = self._calculate_extraction_confidence(total_amount, items)

        return {
            'total_amount': total_amount,
            'items': items,
            'merchant_name': merchant_name,
            'date': date,
            'confidence': confidence
        }

    def _extract_total_amount(self, text: str) -> Optional[float]:
        """Extract total amount from receipt text"""
        # Pattern 1: Look for TOTAL followed by amount
        total_patterns = [
            r'TOTAL\s*[:\-]?\s*\$?(\d+(?:,\d{3})*(?:[,.]\d{2})?)',
            r'AMOUNT\s*[:\-]?\s*\$?(\d+(?:,\d{3})*(?:[,.]\d{2})?)',
            r'GRAND\s+TOTAL\s*[:\-]?\s*\$?(\d+(?:,\d{3})*(?:[,.]\d{2})?)',
            r'SUBTOTAL\s*[:\-]?\s*\$?(\d+(?:,\d{3})*(?:[,.]\d{2})?)',
        ]

        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                # Handle comma decimal separator (e.g., "37,83" -> "37.83")
                amount_str = amount_str.replace(',', '.') if '.' not in amount_str else amount_str
                try:
                    return float(amount_str)
                except ValueError:
                    continue

        # Pattern 2: Amount at end of receipt (common pattern)
        end_patterns = [
            r'(\d+(?:,\d{3})*(?:[,.]\d{2})?)\s*TOTAL',
            r'AMOUNT\s*[:\-]?\s*\$?(\d+(?:,\d{3})*(?:[,.]\d{2})?)',
            r'BALANCE\s*[:\-]?\s*\$?(\d+(?:,\d{3})*(?:[,.]\d{2})?)',
            r'(\d+(?:,\d{3})*(?:[,.]\d{2})?)\s*BALANCE',
        ]

        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1) if match.lastindex >= 1 else match.group(0).replace(',', '')
                # Handle comma decimal separator
                amount_str = amount_str.replace(',', '.') if '.' not in amount_str else amount_str
                try:
                    return float(amount_str)
                except ValueError:
                    continue

        # Pattern 3: Look for amounts before TOTAL
        amount_before_total = r'(\d+(?:,\d{3})*(?:[,.]\d{2})?)\s+(?:TOTAL|AMOUNT|BALANCE)'
        match = re.search(amount_before_total, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '')
            amount_str = amount_str.replace(',', '.') if '.' not in amount_str else amount_str
            try:
                return float(amount_str)
            except ValueError:
                pass

        return None

    def _extract_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract item lines with price patterns"""
        items = []
        lines = text.split('\n')

        for line in lines:
            item = self._parse_item_line(line)
            if item:
                items.append(item)

        return items

    def _parse_item_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single line to extract item information"""
        # Pattern: Description Price [Unit Price] [Quantity]
        # Look for price patterns at the end of lines
        price_patterns = [
            r'(.+?)\s+(\d+\.\d{2})\s*$',  # Description Price
            r'(.+?)\s+(\d+\.\d{2})\s+(\d+\.\d{2})\s*$',  # Description UnitPrice Price
            r'(.+?)\s+(\d+)\s+(\d+\.\d{2})\s*$',  # Description Quantity Price
            r'(.+?)\s+\$?(\d+\.\d{2})\s*$',  # Description $Price
            r'(.+?)\s+(\d+\.\d{2})\s+S?\s*$',  # Description Price S (for savings)
            # Handle space-separated decimals (e.g., "6 . 49")
            r'(.+?)\s+(\d+)\s+\.\s+(\d{2})\s*$',  # Description X . YY
        ]

        for pattern in price_patterns:
            match = re.match(pattern, line.strip(), re.IGNORECASE)
            if match:
                groups = match.groups()

                if len(groups) == 2:
                    # Description and Price
                    description = groups[0].strip()
                    price = self._parse_price(groups[1])
                    return {
                        'description': description,
                        'price': price,
                        'quantity': 1,
                        'unit_price': price
                    }
                elif len(groups) == 3:
                    # Description, UnitPrice, Price OR Description, Quantity, Price
                    description = groups[0].strip()
                    price = self._parse_price(groups[2])

                    # Try to determine if groups[1] is unit price or quantity
                    if '.' in groups[1]:
                        # Likely unit price
                        unit_price = self._parse_price(groups[1])
                        quantity = 1 if unit_price > 0 else price / unit_price
                    else:
                        # Likely quantity
                        quantity = int(groups[1]) if groups[1].isdigit() else 1
                        unit_price = price / quantity if quantity > 0 else price

                    return {
                        'description': description,
                        'price': price,
                        'quantity': quantity,
                        'unit_price': unit_price
                    }

        return None

    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float"""
        try:
            # Remove currency symbols and spaces
            clean_price = re.sub(r'[^\d.]', '', price_str)
            # Handle space-separated decimals (e.g., "6 . 49" -> "6.49")
            clean_price = clean_price.replace(' ', '')
            return float(clean_price)
        except (ValueError, AttributeError):
            return 0.0

    def _extract_merchant_name(self, text: str) -> Optional[str]:
        """Extract merchant name using common patterns"""
        # Look for common store names at the beginning
        merchant_patterns = [
            r'^(GROCERY|PRODUCE|MARKET|STORE|SHOP|PHARMACY|WALMART|TARGET|COSTCO|SAFEWAY)',
            r'^(YOUR CASHIER TODAY WAS|CASHIER)\s+(.+?)(?:\s+\d+)',
            r'^(.+?)\s+(GROCERY|MARKET|STORE)',
        ]

        for pattern in merchant_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date using common patterns"""
        date_patterns = [
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',  # MM/DD/YY or MM-DD-YY
            r'(\d{1,2}\/\d{1,2}\/\d{2,4})',  # MM/DD/YYYY
            r'(\d{2,4}[\/\-]\d{1,2}[\/\-]\d{1,2})',  # YYYY-MM-DD
            r'(\w{3,9}\s+\d{1,2},?\s+\d{4})',  # Month DD, YYYY
            r'(\d{1,2}\s+\w{3,9}\s+\d{4})',  # DD Month YYYY
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    def _calculate_extraction_confidence(self, total_amount: Optional[float], items: List[Dict]) -> float:
        """Calculate confidence score for field extraction"""
        confidence = 0.0

        # Base confidence
        confidence += 0.3

        # Total amount found
        if total_amount and total_amount > 0:
            confidence += 0.3

        # Items extracted
        if items:
            confidence += min(0.4, len(items) * 0.1)  # Up to 0.4 for items

        return min(confidence, 1.0)

    def score_ocr_quality(self, text: str, detailed: bool = False, debug: bool = False) -> float:
        """
        Score OCR output quality using deterministic logic and regex patterns.

        Scoring Criteria:
        - Length of text (0-20 points)
        - Presence of price patterns like 3.99 (0-25 points)
        - Presence of "TOTAL" keyword (0-20 points)
        - Ratio of valid words vs noisy tokens (0-25 points)
        - Penalty for excessive symbols/noise (0-10 points)

        Args:
            text: OCR text to evaluate
            detailed: If True, return dict with component scores instead of just total score
            debug: If True, print detailed scoring breakdown

        Returns:
            float: Quality score between 0.0 and 1.0 (if detailed=False)
            dict: Detailed scoring breakdown (if detailed=True)
        """
        if not text or not text.strip():
            if detailed:
                return {
                    'total_score': 0.0,
                    'component_scores': {
                        'text_length': 0.0,
                        'price_patterns': 0.0,
                        'total_keyword': 0.0,
                        'word_quality': 0.0,
                        'noise_penalty': 0.0
                    },
                    'raw_scores': {
                        'text_length_points': 0.0,
                        'price_patterns_points': 0.0,
                        'total_keyword_points': 0.0,
                        'word_quality_points': 0.0,
                        'noise_penalty_points': 0.0
                    },
                    'max_scores': {
                        'text_length': 20.0,
                        'price_patterns': 25.0,
                        'total_keyword': 20.0,
                        'word_quality': 25.0,
                        'noise_penalty': 10.0
                    },
                    'text_stats': {
                        'length': len(text.strip()) if text else 0,
                        'word_count': len(text.split()) if text and text.strip() else 0,
                        'line_count': len(text.split('\n')) if text and text.strip() else 0
                    }
                }
            return 0.0

        # Calculate component scores
        length_score = self._score_text_length(text)
        price_score = self._score_price_patterns(text)
        total_score = self._score_total_keyword(text)
        word_quality_score = self._score_word_quality(text)
        noise_penalty = self._calculate_noise_penalty(text)

        # Calculate total score
        raw_total = length_score + price_score + total_score + word_quality_score - noise_penalty
        final_score = max(0.0, min(1.0, raw_total / 100.0))  # Normalize to 0-1

        # Calibration adjustments
        price_score_norm = price_score / 25.0
        total_score_norm = total_score / 20.0

        # Hard rejection: non-receipt text
        if total_score_norm == 0 and price_score_norm < 0.3:
            final_score = min(final_score, 0.4)

        # Stronger rejection for structured non-receipt text
        if total_score_norm == 0 and price_score_norm < 0.2 and len(text.strip()) > 50:
            final_score = min(final_score, 0.35)

        # Noise clamp
        if noise_penalty >= 4.0:
            final_score = min(final_score, 0.3)

        # Keywords without prices clamp (descriptive text with receipt words)
        if total_score_norm > 0 and price_score_norm < 0.2 and len(text.strip()) > 100:
            final_score = min(final_score, 0.4)

        # Strengthen medium text floor
        if final_score < 0.3 and price_score_norm >= 0.3:
            final_score = 0.3

        # Boost structured receipts
        if total_score_norm > 0 and price_score_norm > 0.3:
            final_score = max(final_score, 0.5)

        # Clamp final
        final_score = max(0.0, min(1.0, final_score))

        # Debug output
        if debug:
            self._print_scoring_debug(
                text, length_score, price_score, total_score,
                word_quality_score, noise_penalty, final_score
            )

        # Return detailed breakdown if requested
        if detailed:
            return {
                'total_score': final_score,
                'component_scores': {
                    'text_length': length_score / 20.0,      # Normalized to 0-1
                    'price_patterns': price_score / 25.0,   # Normalized to 0-1
                    'total_keyword': total_score / 20.0,    # Normalized to 0-1
                    'word_quality': word_quality_score / 25.0,  # Normalized to 0-1
                    'noise_penalty': noise_penalty / 10.0   # Normalized to 0-1
                },
                'raw_scores': {
                    'text_length_points': length_score,
                    'price_patterns_points': price_score,
                    'total_keyword_points': total_score,
                    'word_quality_points': word_quality_score,
                    'noise_penalty_points': noise_penalty
                },
                'max_scores': {
                    'text_length': 20.0,
                    'price_patterns': 25.0,
                    'total_keyword': 20.0,
                    'word_quality': 25.0,
                    'noise_penalty': 10.0
                },
                'text_stats': {
                    'length': len(text.strip()),
                    'word_count': len(text.split()),
                    'line_count': len(text.split('\n'))
                }
            }

        return final_score

    def _score_text_length(self, text: str) -> float:
        """Score text length (0-20 points)"""
        return min(len(text.strip()) / 10, 20)

    def _score_price_patterns(self, text: str) -> float:
        """Score price patterns (0-25 points)"""
        # Match standard price formats: 3.99, $4.99, etc.
        price_patterns = r'\$?\d+\.\d{2}'
        price_matches = len(re.findall(price_patterns, text))
        # Match space-separated prices like "3 99" but not "123 456" sequences
        # Only match 1-2 digits before space (typical for dollar amounts), exactly 2 digits after
        space_prices = len(re.findall(r'(?<!\d)\d{1,2}\s+\d{2}(?!\d)', text))
        total_matches = price_matches + (space_prices * 0.5)
        return min(total_matches * 5, 25)

    def _score_total_keyword(self, text: str) -> float:
        """Score total keywords (0-20 points)"""
        total_keywords = ['total', 'amount', 'sum', 'subtotal']
        text_lower = text.lower()
        total_count = sum(1 for keyword in total_keywords if keyword in text_lower)
        # Boost weight: 7 points per keyword to ensure >= 0.6 with 2 keywords
        return min(total_count * 7, 20)

    def _score_word_quality(self, text: str) -> float:
        """Score word quality (0-25 points)"""
        words = text.split()
        if not words:
            return 0
        valid_words = sum(1 for word in words if self._is_likely_valid_word(word))
        # Penalize repetitive words (like "aaaaaaaa" or "x x x x x")
        unique_words = len(set(words))
        total_words = len(words)
        # Check for repetitive patterns
        is_repetitive = unique_words == 1 and (total_words > 5 or (total_words == 1 and len(words[0]) > 10))
        if is_repetitive:
            return 5  # Low score for repetitive text
        return (valid_words / total_words * 25)

    def _calculate_noise_penalty(self, text: str) -> float:
        """Calculate noise penalty (0-10 points)"""
        noise_chars = len(re.findall(r'[^\w\s$.,%\-:\/\n]', text))
        text_len = len(text.strip())
        noise_ratio = noise_chars / text_len if text_len > 0 else 0
        # Increase noise penalty threshold
        if noise_ratio > 0.3:
            return 4.0  # 0.4 normalized = 4.0 points
        return 0.0

    def _is_likely_valid_word(self, word: str) -> bool:
        """Simple heuristic to check if a word is likely valid"""
        # This is a simplified check - in practice, you might use a dictionary
        # or more sophisticated language model
        return len(word) > 1 and not word.isdigit()

    def _print_scoring_debug(self, text: str, length_score: float, price_score: float,
                           total_score: float, word_quality_score: float,
                           noise_penalty: float, final_score: float):
        """Print detailed scoring breakdown for debugging"""
        logger.debug("=== OCR QUALITY SCORING DEBUG ===")
        logger.debug(f"Text length: {len(text)} chars")
        logger.debug(f"Length score: {length_score:.1f}/20")
        logger.debug(f"Price patterns score: {price_score:.1f}/25")
        logger.debug(f"Total keyword score: {total_score:.1f}/20")
        logger.debug(f"Word quality score: {word_quality_score:.1f}/25")
        logger.debug(f"Noise penalty: -{noise_penalty:.1f}/10")
        logger.debug(f"Raw total: {length_score + price_score + total_score + word_quality_score - noise_penalty:.1f}/100")
        logger.debug(f"Final score: {final_score:.3f}")
        logger.debug("================================")

    def should_fallback(self, text: str, threshold: float) -> bool:
        """Determine if OCR quality is below threshold and fallback is needed"""
        quality_score = self.score_ocr_quality(text)
        return quality_score < threshold

    def get_fallback_reasoning(self, text: str, threshold: float) -> Dict[str, Any]:
        """Get detailed reasoning for fallback decision"""
        quality_score = self.score_ocr_quality(text, detailed=True)

        # Determine reasoning
        reasoning_parts = []

        if quality_score['component_scores']['text_length'] < 0.5:
            reasoning_parts.append(f"Text too short ({len(text)} chars)")

        if quality_score['component_scores']['price_patterns'] < 0.6:
            reasoning_parts.append("Insufficient price patterns detected")

        if quality_score['component_scores']['total_keyword'] < 0.5:
            reasoning_parts.append("Missing TOTAL/AMOUNT keywords")

        if quality_score['component_scores']['word_quality'] < 0.6:
            reasoning_parts.append("Low word quality (many noisy tokens)")

        if quality_score['component_scores']['noise_penalty'] > 0.5:
            reasoning_parts.append("Excessive special characters/noise")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "General quality concerns"

        return {
            'should_fallback': quality_score['total_score'] < threshold,
            'quality_score': quality_score['total_score'],
            'threshold': threshold,
            'reasoning': reasoning,
            'component_scores': quality_score['component_scores'],
            'recommendation': 'FALLBACK' if quality_score['total_score'] < threshold else 'PROCEED'
        }
