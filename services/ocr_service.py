"""OCR service for extracting text from images using local EasyOCR"""

import logging
import re
import os
from typing import Any, List, Dict
from langchain_core.runnables import RunnableLambda
import easyocr
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class OCRService:
    """Service for OCR text extraction using local EasyOCR"""

    def __init__(self, use_gpu: bool = False, lang: List[str] = ['en'], confidence_threshold: float = 0.7, debug: bool = False):
        """
        Initialize EasyOCR service

        Args:
            use_gpu: Whether to use GPU acceleration
            lang: List of language codes for OCR (e.g., ['en'], ['en', 'ch'])
            confidence_threshold: Minimum confidence score for text extraction (0.0-1.0)
            debug: Enable debug logging for OCR pipeline stages
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))  # Clamp to valid range
        self.debug = debug

        # Initialize EasyOCR with specified languages
        self.ocr = easyocr.Reader(lang, gpu=use_gpu)

        self._preprocess_chain = RunnableLambda(self._preprocess_image)
        self._ocr_chain = RunnableLambda(self._extract_easyocr_text)

        logger.info(f"Initialized OCRService with EasyOCR (GPU: {use_gpu}, Lang: {lang}, Confidence: {self.confidence_threshold}, Debug: {debug})")

    def preprocess_image(self, image_path: str) -> Any:
        """Preprocess image for OCR"""
        return self._preprocess_image(image_path)

    def extract_text(self, image_path: str) -> str:
        """Extract text from image using EasyOCR"""
        return self._extract_easyocr_text(image_path)

    def _preprocess_image(self, image_path: str) -> dict:
        """Preprocess image for EasyOCR"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load and validate image
            image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to numpy array for EasyOCR
            image_array = np.array(image)

            return {
                "image_path": image_path,
                "image_array": image_array,
                "original_size": image.size
            }

        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise

    def _extract_easyocr_text(self, image_input: Any) -> str:
        """Extract text using EasyOCR"""
        if isinstance(image_input, str):
            # If it's a string path, preprocess first
            processed = self._preprocess_image(image_input)
        else:
            processed = image_input

        try:
            # Run EasyOCR
            results = self.ocr.readtext(processed['image_array'])

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
            logger.error(f"EasyOCR error for {processed['image_path']}: {str(e)}")
            raise

    def _combine_text_lines(self, text_lines: List[str], ocr_results: List) -> str:
        """Combine OCR text lines preserving layout structure"""
        if not text_lines or not ocr_results:
            return ""

        # Create list of (y_coord, x_coord, text) for proper sorting
        lines_with_coords = []

        # Map text_lines to their corresponding OCR results
        text_index = 0
        for bbox, text, confidence in ocr_results:
            # Only include results that passed the confidence filter
            if text and confidence >= self.confidence_threshold and text_index < len(text_lines):
                # Use top-left corner coordinates
                y_coord = bbox[0][1] if bbox else 0
                x_coord = bbox[0][0] if bbox else 0

                lines_with_coords.append((y_coord, x_coord, text_lines[text_index]))
                text_index += 1

        # Sort by y-coordinate first (top-to-bottom), then x-coordinate (left-to-right)
        # This maintains stable ordering for items on the same line
        lines_with_coords.sort(key=lambda x: (x[0], x[1]))

        # Extract sorted text lines
        sorted_lines = [line[2] for line in lines_with_coords]

        # Join lines with newlines, preserving structure
        combined_text = '\n'.join(sorted_lines)

        return combined_text

    def _normalize_text(self, text: str) -> str:
        """Apply basic text normalization for better parsing"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common OCR errors
        text = text.replace('O', '0')  # Replace letter O with zero in numeric contexts
        text = text.replace('l', '1')  # Replace letter l with one in numeric contexts
        text = text.replace('I', '1')  # Replace letter I with one in numeric contexts

        # Fix common receipt-specific OCR errors
        text = text.replace('S', '$')  # Replace S with dollar sign in money contexts
        text = text.replace('s', '$')  # Replace s with dollar sign in money contexts

        # Clean up special characters but keep important ones
        text = re.sub(r'[^\w\s$.,%\-:\/\n]', '', text)

        # Ensure proper spacing around punctuation
        text = re.sub(r'([.,%])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def get_ocr_confidence(self, image_path: str) -> float:
        """Get average confidence score for OCR extraction"""
        try:
            processed = self._preprocess_image(image_path)
            results = self.ocr.readtext(processed['image_array'])

            confidence_scores = []
            if results:
                for (bbox, text, confidence) in results:
                    if text and confidence > 0:
                        confidence_scores.append(confidence)

            return np.mean(confidence_scores) if confidence_scores else 0.0

        except Exception as e:
            logger.error(f"Error getting OCR confidence for {image_path}: {str(e)}")
            return 0.0

    def extract_text_with_confidence(self, image_path: str) -> Dict[str, Any]:
        """Extract text with detailed confidence information"""
        try:
            processed = self._preprocess_image(image_path)
            results = self.ocr.readtext(processed['image_array'])

            text_lines = []
            confidence_scores = []
            detailed_results = []

            if results:
                for (bbox, text, confidence) in results:
                    if text and confidence >= self.confidence_threshold:
                        text_lines.append(text.strip())
                        confidence_scores.append(confidence)

                        detailed_results.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'bbox': bbox
                        })

            # Combine and normalize text
            combined_text = self._combine_text_lines(text_lines, results)
            normalized_text = self._normalize_text(combined_text)

            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0

            return {
                'text': normalized_text,
                'raw_text': combined_text,
                'average_confidence': avg_confidence,
                'line_count': len(text_lines),
                'detailed_results': detailed_results
            }

        except Exception as e:
            logger.error(f"Error in extract_text_with_confidence for {image_path}: {str(e)}")
            raise
