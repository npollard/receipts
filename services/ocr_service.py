"""OCR service for extracting text from images using local EasyOCR"""

import logging
import re
import os
from typing import Any, List, Dict
from langchain_core.runnables import RunnableLambda
import easyocr
from PIL import Image
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from utils.file_utils import encode_file_base64

logger = logging.getLogger(__name__)


class OCRService:
    """Service for OCR text extraction using local EasyOCR"""

    def __init__(self, use_gpu: bool = False, lang: List[str] = ['en'], confidence_threshold: float = 0.7, debug: bool = False, comparison_mode: bool = False):
        """
        Initialize EasyOCR service

        Args:
            use_gpu: Whether to use GPU acceleration
            lang: List of language codes for OCR (e.g., ['en'], ['en', 'ch'])
            confidence_threshold: Minimum confidence score for text extraction (0.0-1.0)
            debug: Enable debug logging for OCR pipeline stages
            comparison_mode: Enable comparison with OpenAI Vision OCR for evaluation
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))  # Clamp to valid range
        self.debug = debug
        self.comparison_mode = comparison_mode

        # Initialize EasyOCR with specified languages
        self.ocr = easyocr.Reader(lang, gpu=use_gpu)

        # Initialize OpenAI Vision client for comparison mode
        if comparison_mode:
            self.vision_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                api_key=os.environ.get("OPENAI_API_KEY")
            )

        self._preprocess_chain = RunnableLambda(self._preprocess_image)
        self._ocr_chain = RunnableLambda(self._extract_easyocr_text)

        logger.info(f"Initialized OCRService with EasyOCR (GPU: {use_gpu}, Lang: {lang}, Confidence: {self.confidence_threshold}, Debug: {debug}, Comparison: {comparison_mode})")

    def preprocess_image(self, image_path: str) -> Any:
        """Preprocess image for OCR"""
        return self._preprocess_image(image_path)

    def extract_text(self, image_path: str) -> str:
        """Extract text from image using EasyOCR"""
        if self.comparison_mode:
            return self._extract_with_comparison(image_path)
        else:
            return self._extract_easyocr_text(image_path)

    def _extract_with_comparison(self, image_path: str) -> str:
        """Extract text using both EasyOCR and OpenAI Vision for comparison"""
        try:
            # Get EasyOCR result
            easyocr_result = self._extract_easyocr_text(image_path)

            # Get OpenAI Vision result
            vision_result = self._extract_vision_text(image_path)

            # Print comparison
            self._print_comparison(image_path, easyocr_result, vision_result)

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

    def _print_comparison(self, image_path: str, easyocr_text: str, vision_text: str):
        """Print comparison between EasyOCR and OpenAI Vision results"""
        print(f"\n{'='*80}")
        print(f"OCR COMPARISON: {image_path}")
        print(f"{'='*80}")

        print(f"\n{'-'*60}")
        print("EASYOCR RESULT:")
        print(f"{'-'*60}")
        print(f"Length: {len(easyocr_text)} characters")
        print(f"Text:\n{easyocr_text}")

        print(f"\n{'-'*60}")
        print("OPENAI VISION RESULT:")
        print(f"{'-'*60}")
        print(f"Length: {len(vision_text)} characters")
        print(f"Text:\n{vision_text}")

        print(f"\n{'-'*60}")
        print("COMPARISON SUMMARY:")
        print(f"{'-'*60}")
        print(f"EasyOCR length:  {len(easyocr_text)} chars")
        print(f"Vision length:  {len(vision_text)} chars")
        print(f"Length diff:    {len(easyocr_text) - len(vision_text)} chars")

        # Calculate word overlap
        easyocr_words = set(easyocr_text.lower().split())
        vision_words = set(vision_text.lower().split())
        common_words = easyocr_words.intersection(vision_words)

        print(f"EasyOCR words:  {len(easyocr_words)}")
        print(f"Vision words:   {len(vision_words)}")
        print(f"Common words:   {len(common_words)}")
        print(f"Word overlap:   {len(common_words) / max(len(easyocr_words), len(vision_words)) * 100:.1f}%")

        print(f"\n{'='*80}\n")

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
        """Apply context-aware text normalization for better parsing"""
        if not text:
            return ""

        # Remove excessive whitespace first
        text = re.sub(r'\s+', ' ', text)

        # Apply context-aware character corrections
        text = self._fix_character_errors(text)

        # Clean up special characters but keep important ones
        text = re.sub(r'[^\w\s$.,%\-:\/\n]', '', text)

        # Ensure proper spacing around punctuation
        text = re.sub(r'([.,%])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

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

        # Remove punctuation for analysis, but preserve it for reconstruction
        prefix_punct = ''
        suffix_punct = ''
        clean_word = word

        # Extract leading punctuation
        while clean_word and not clean_word[0].isalnum():
            prefix_punct += clean_word[0]
            clean_word = clean_word[1:]

        # Extract trailing punctuation
        while clean_word and not clean_word[-1].isalnum():
            suffix_punct = clean_word[-1] + suffix_punct
            clean_word = clean_word[:-1]

        # Apply context-aware corrections to the clean word
        corrected_clean_word = self._apply_context_corrections(clean_word)

        # Reconstruct word with original punctuation
        return prefix_punct + corrected_clean_word + suffix_punct

    def _apply_context_corrections(self, word: str) -> str:
        """Apply context-aware character corrections to a clean word"""
        if not word:
            return word

        corrected = word

        # Rule 1: Fix 0/O confusion based on context
        # If word is mostly numeric, O should be 0
        # If word is mostly alphabetic, 0 should be O
        digit_count = sum(1 for c in word if c.isdigit())
        alpha_count = sum(1 for c in word if c.isalpha())

        if digit_count > alpha_count and len(word) > 1:
            # Mostly numeric - replace O with 0
            corrected = corrected.replace('O', '0')
        elif alpha_count > digit_count:
            # Mostly alphabetic - replace 0 with O
            corrected = corrected.replace('0', 'O')

        # Rule 2: Fix l/1/I confusion in numeric contexts
        if digit_count > alpha_count:
            # Numeric context - replace l and I with 1
            corrected = corrected.replace('l', '1').replace('I', '1')
        elif alpha_count > digit_count:
            # Alphabetic context - replace 1 with l (more common than I in words)
            corrected = corrected.replace('1', 'l')

        # Rule 3: Fix S/$ confusion based on money context
        # Use heuristics to detect money amounts
        if self._is_money_context(word):
            # Money context - S should be $
            corrected = corrected.replace('S', '$').replace('s', '$')
        elif self._looks_like_money_amount(word):
            # Money amount context - S should be $
            corrected = corrected.replace('S', '$').replace('s', '$')
        else:
            # Non-money context - $ should be S
            corrected = corrected.replace('$', 'S')

        # Rule 4: Specific receipt word corrections
        corrected = self._fix_receipt_words(corrected)

        return corrected

    def _is_money_context(self, word: str) -> bool:
        """Check if word is in a money context"""
        money_indicators = ['$', 'price', 'cost', 'total', 'amount', 'pay', 'cash', 'card', 'debit', 'credit']
        return any(indicator in word.lower() for indicator in money_indicators)

    def _looks_like_money_amount(self, word: str) -> bool:
        """Check if word looks like a money amount"""
        # Pattern: digits with optional decimal, possibly with S/$ prefix
        money_pattern = r'^[S$]?\d+\.?\d*$'
        return bool(re.match(money_pattern, word))

    def _fix_receipt_words(self, word: str) -> str:
        """Fix specific common OCR errors in receipt-related words"""
        # Common receipt word corrections
        corrections = {
            'GR0CERY': 'GROCERY',
            'PR0DUCE': 'PRODUCE',
            'DE$CHUTE$': 'DESCHUTES',
            'L1QU0R': 'LIQUOR',
            'CA$H': 'CASH',
            'T0TAL': 'TOTAL',
            'AM0UNT': 'AMOUNT',
            'BALANCE': 'BALANCE',
            'PURCHA$E': 'PURCHASE',
            'CARD': 'CARD',
            'DEBI': 'DEBI',
            'PR1MARY': 'PRIMARY',
            'AUTH': 'AUTH',
            'NF$': 'NFS',  # No Fructose Sugar
            'BLACKGPK': 'BLACK 6PK',  # Common beer packaging
        }

        # Apply corrections if exact match
        if word in corrections:
            return corrections[word]

        # Apply partial corrections for common patterns
        if 'GR0CERY' in word:
            word = word.replace('GR0CERY', 'GROCERY')
        if 'PR0DUCE' in word:
            word = word.replace('PR0DUCE', 'PRODUCE')
        if 'L1QU0R' in word:
            word = word.replace('L1QU0R', 'LIQUOR')
        if 'CA$H' in word:
            word = word.replace('CA$H', 'CASH')
        if 'T0TAL' in word:
            word = word.replace('T0TAL', 'TOTAL')
        if 'AM0UNT' in word:
            word = word.replace('AM0UNT', 'AMOUNT')

        return word

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
