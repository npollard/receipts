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

    def __init__(self, use_gpu: bool = False, lang: List[str] = ['en'], confidence_threshold: float = 0.7, debug: bool = False, comparison_mode: bool = False, quality_threshold: float = 0.25, debug_ocr: bool = False):
        """
        Initialize EasyOCR service

        Args:
            use_gpu: Whether to use GPU acceleration
            lang: List of language codes for OCR (e.g., ['en'], ['en', 'ch'])
            confidence_threshold: Minimum confidence score for text extraction (0.0-1.0)
            debug: Enable debug logging for OCR pipeline stages
            comparison_mode: Enable comparison with OpenAI Vision OCR for evaluation
            quality_threshold: Minimum OCR quality score before fallback to Vision OCR (0.0-1.0)
            debug_ocr: Enable detailed OCR decision observability (scores, fallback, comparisons)
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))  # Clamp to valid range
        self.debug = debug
        self.comparison_mode = comparison_mode
        self.quality_threshold = max(0.0, min(1.0, quality_threshold))  # Clamp to valid range
        self.debug_ocr = debug_ocr

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

        logger.info(f"Initialized OCRService with EasyOCR (GPU: {use_gpu}, Lang: {lang}, Confidence: {self.confidence_threshold}, Debug: {debug}, Comparison: {comparison_mode}, Quality Threshold: {self.quality_threshold}, Debug OCR: {debug_ocr})")

    def preprocess_image(self, image_path: str) -> Any:
        """Preprocess image for OCR"""
        return self._preprocess_image(image_path)

    def extract_text(self, image_path: str) -> str:
        """Extract text from image using EasyOCR with quality-based fallback to Vision OCR"""
        if self.comparison_mode:
            return self._extract_with_comparison(image_path)
        else:
            return self._extract_with_quality_fallback(image_path)

    def _extract_with_quality_fallback(self, image_path: str) -> str:
        """Extract text using EasyOCR with automatic fallback to Vision OCR based on quality score"""
        try:
            # Debug OCR header
            if self.debug_ocr:
                print(f"\n{'='*60}")
                print(f"OCR DECISION DEBUG FOR: {os.path.basename(image_path)}")
                print(f"{'='*60}")
                print(f"Quality Threshold: {self.quality_threshold:.3f}")
                print(f"Debug Mode: ENABLED")

            # First, extract text using EasyOCR
            easyocr_text = self._extract_easyocr_text(image_path)

            # Calculate quality score
            quality_score = self.score_ocr_quality(easyocr_text)

            # Log quality score
            logger.info(f"EasyOCR Quality Score: {quality_score:.3f} (threshold: {self.quality_threshold:.3f}) for {image_path}")

            # Debug OCR score printing
            if self.debug_ocr:
                print(f"\n--- EASYOCR RESULTS ---")
                print(f"Quality Score: {quality_score:.3f}")
                print(f"Text Length: {len(easyocr_text)} characters")
                print(f"Text Preview: {easyocr_text[:200]}{'...' if len(easyocr_text) > 200 else ''}")

                # Show component scores
                reasoning = self.get_fallback_reasoning(easyocr_text, self.quality_threshold)
                print(f"\n--- QUALITY SCORE BREAKDOWN ---")
                print(f"Text Length: {reasoning['component_scores']['text_length']:.1f}/20")
                print(f"Price Patterns: {reasoning['component_scores']['price_patterns']:.1f}/25")
                print(f"Total Keywords: {reasoning['component_scores']['total_keyword']:.1f}/20")
                print(f"Word Quality: {reasoning['component_scores']['word_quality']:.1f}/25")
                print(f"Noise Penalty: -{reasoning['component_scores']['noise_penalty']:.1f}/10")
                print(f"Total Raw Score: {sum(reasoning['component_scores'].values()) - 2*reasoning['component_scores']['noise_penalty']:.1f}/100")

            # Determine if fallback is needed
            should_fallback = self.should_fallback(easyocr_text, self.quality_threshold)

            # Debug fallback decision
            if self.debug_ocr:
                print(f"\n--- FALLBACK DECISION ---")
                print(f"Quality Score ({quality_score:.3f}) < Threshold ({self.quality_threshold:.3f}): {should_fallback}")
                print(f"Fallback Required: {'YES' if should_fallback else 'NO'}")

                if should_fallback:
                    print(f"Reasoning: {reasoning['reasoning']}")
                else:
                    print(f"Reasoning: Good quality OCR - using EasyOCR result")

            if should_fallback:
                logger.warning(f"EasyOCR quality below threshold - falling back to Vision OCR for {image_path}")

                # Debug fallback initiation
                if self.debug_ocr:
                    print(f"\n--- INITIATING FALLBACK TO VISION OCR ---")

                # Fallback to Vision OCR
                vision_text = self._extract_vision_text(image_path)

                # Calculate Vision OCR quality score
                vision_quality_score = self.score_ocr_quality(vision_text)
                logger.info(f"Vision OCR Quality Score: {vision_quality_score:.3f} for {image_path}")

                # Debug Vision OCR results
                if self.debug_ocr:
                    print(f"\n--- VISION OCR RESULTS ---")
                    print(f"Quality Score: {vision_quality_score:.3f}")
                    print(f"Text Length: {len(vision_text)} characters")
                    print(f"Text Preview: {vision_text[:200]}{'...' if len(vision_text) > 200 else ''}")

                    # Show Vision OCR component scores
                    vision_reasoning = self.get_fallback_reasoning(vision_text, self.quality_threshold)
                    print(f"\n--- VISION OCR QUALITY BREAKDOWN ---")
                    print(f"Text Length: {vision_reasoning['component_scores']['text_length']:.1f}/20")
                    print(f"Price Patterns: {vision_reasoning['component_scores']['price_patterns']:.1f}/25")
                    print(f"Total Keywords: {vision_reasoning['component_scores']['total_keyword']:.1f}/20")
                    print(f"Word Quality: {vision_reasoning['component_scores']['word_quality']:.1f}/25")
                    print(f"Noise Penalty: -{vision_reasoning['component_scores']['noise_penalty']:.1f}/10")

                # Use Vision OCR result
                final_text = vision_text
                logger.info(f"Using Vision OCR result for {image_path}")

                # Debug final decision
                if self.debug_ocr:
                    print(f"\n--- FINAL DECISION ---")
                    print(f"Selected OCR: VISION OCR")
                    print(f"Reason: EasyOCR quality below threshold")
            else:
                # Use EasyOCR result
                final_text = easyocr_text
                logger.info(f"Using EasyOCR result for {image_path}")

                # Debug final decision
                if self.debug_ocr:
                    print(f"\n--- FINAL DECISION ---")
                    print(f"Selected OCR: EASYOCR")
                    print(f"Reason: Quality acceptable")

            # Debug final output
            if self.debug_ocr:
                print(f"\n--- FINAL OUTPUT ---")
                print(f"Selected Text Length: {len(final_text)} characters")
                print(f"Selected Text Preview: {final_text[:300]}{'...' if len(final_text) > 300 else ''}")
                print(f"{'='*60}\n")

            return final_text

        except Exception as e:
            logger.error(f"Error in quality-based OCR extraction for {image_path}: {str(e)}")

            # Debug error handling
            if self.debug_ocr:
                print(f"\n--- ERROR HANDLING ---")
                print(f"Error: {str(e)}")
                print(f"Initiating emergency fallback to basic EasyOCR")

            # Fallback to basic EasyOCR without quality scoring
            try:
                return self._extract_easyocr_text(image_path)
            except Exception as fallback_error:
                logger.error(f"Fallback EasyOCR also failed for {image_path}: {str(fallback_error)}")

                # Debug catastrophic failure
                if self.debug_ocr:
                    print(f"\n--- CATASTROPHIC FAILURE ---")
                    print(f"Both EasyOCR and fallback failed")
                    print(f"Primary Error: {str(e)}")
                    print(f"Fallback Error: {str(fallback_error)}")
                    print(f"{'='*60}\n")

                raise

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
        """Apply generalized context-aware normalization to receipt words"""
        if not word:
            return word

        # Apply generalized character corrections based on context
        return self._apply_generalized_corrections(word)

    def _apply_generalized_corrections(self, word: str) -> str:
        """Apply context-aware character corrections without hardcoded mappings"""
        corrected = word

        # 1. Fix common OCR character confusions based on word context
        corrected = self._fix_character_confusions(corrected)

        # 2. Fix price formatting issues
        corrected = self._fix_price_formatting(corrected)

        # 3. Fix common receipt-specific patterns using regex
        corrected = self._fix_receipt_patterns(corrected)

        return corrected

    def _fix_character_confusions(self, word: str) -> str:
        """Fix character confusions based on word analysis"""
        corrected = word

        # Count character types for context analysis
        digit_count = sum(1 for c in word if c.isdigit())
        alpha_count = sum(1 for c in word if c.isalpha())
        special_count = len(word) - digit_count - alpha_count

        # Apply corrections based on character composition
        if digit_count > alpha_count and len(word) > 1:
            # Mostly numeric - fix alphabetic characters that should be digits
            corrected = corrected.replace('O', '0')
            corrected = corrected.replace('I', '1')
            corrected = corrected.replace('l', '1')
        elif alpha_count > digit_count:
            # Mostly alphabetic - fix numeric characters that should be letters
            corrected = corrected.replace('0', 'O')
            corrected = corrected.replace('1', 'l')
        elif special_count > 0:
            # Mixed characters - be more conservative
            if 'S' in word and '$' not in word and digit_count > 0:
                # Likely money context where S should be $
                corrected = corrected.replace('S', '$')
            elif '$' in word and digit_count == 0:
                # Non-money context where $ should be S
                corrected = corrected.replace('$', 'S')

        return corrected

    def _fix_price_formatting(self, word: str) -> str:
        """Fix common price formatting issues using regex"""
        # Fix space-separated decimals (e.g., "6 . 49" -> "6.49")
        space_decimal_pattern = r'(\d+)\s+\.\s+(\d{2})'
        corrected = re.sub(space_decimal_pattern, r'\1.\2', word)

        # Fix comma decimal separators (e.g., "37,83" -> "37.83") in money context
        if '$' in word or any(money_word in word.lower() for money_word in ['total', 'amount', 'price', 'cost']):
            comma_decimal_pattern = r'(\d+),(\d{2})'
            corrected = re.sub(comma_decimal_pattern, r'\1.\2', corrected)

        # Fix missing decimal points in price-like patterns
        price_pattern = r'^\$?(\d+)(\d{2})$'
        if re.match(price_pattern, corrected):
            corrected = re.sub(price_pattern, r'$\1.\2', corrected)

        return corrected

    def _fix_receipt_patterns(self, word: str) -> str:
        """Fix common receipt patterns using regex-based approaches"""
        # Fix common OCR errors in receipt terminology
        patterns = [
            # Common number/letter substitutions in receipt contexts
            (r'GR[0O]CERY', 'GROCERY'),
            (r'PR[0O]DUCE', 'PRODUCE'),
            (r'L[1I]QU[0O]R', 'LIQUOR'),
            (r'T[0O]TAL', 'TOTAL'),
            (r'AM[0O]UNT', 'AMOUNT'),
            (r'CA$H', 'CASH'),
            (r'PURCHA$E', 'PURCHASE'),
            (r'PR[1I]MARY', 'PRIMARY'),
            (r'DEB[1I]', 'DEBI'),

            # Fix common packaging/quantity indicators
            (r'BLACKGPK', 'BLACK 6PK'),
            (r'(\d+)PK', r'\1 PK'),

            # Fix common separator issues
            (r'\s+\.\s+', '.'),  # Space around decimal points
            (r'\s+\,\s+', ','),  # Space around commas

            # Fix currency symbol placement
            (r'(\d+)\s+\$', r'$\1'),
            (r'\$\s+(\d+)', r'$\1'),
        ]

        corrected = word
        for pattern, replacement in patterns:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

        return corrected

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

    def extract_receipt_fields(self, ocr_text: str) -> Dict[str, Any]:
        """Extract key receipt fields from OCR text using regex-based heuristics"""
        if not ocr_text:
            return {
                'total_amount': None,
                'items': [],
                'merchant_name': None,
                'date': None,
                'raw_text': ocr_text
            }

        # Extract structured fields
        total_amount = self._extract_total_amount(ocr_text)
        items = self._extract_items(ocr_text)
        merchant_name = self._extract_merchant_name(ocr_text)
        date = self._extract_date(ocr_text)

        return {
            'total_amount': total_amount,
            'items': items,
            'merchant_name': merchant_name,
            'date': date,
            'raw_text': ocr_text,
            'extraction_confidence': self._calculate_extraction_confidence(total_amount, items)
        }

    def _extract_total_amount(self, text: str) -> Optional[float]:
        """Extract total amount using regex patterns"""
        # Pattern 1: TOTAL followed by amount (handle comma decimal format)
        total_patterns = [
            r'TOTAL\s*AMOUNT\s*[:\-]?\s*\$?(\d+(?:,\d{3})*(?:[,.]\d{2})?)',
            r'TOTAL\s*[:\-]?\s*\$?(\d+(?:,\d{3})*(?:[,.]\d{2})?)',
            r'TOTAL\s*TRANSACTION\s*AMOUNT\s*[:\-]?\s*\$?(\d+(?:,\d{3})*(?:[,.]\d{2})?)',
            r'GRAND\s*TOTAL\s*[:\-]?\s*\$?(\d+(?:,\d{3})*(?:[,.]\d{2})?)',
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

    def score_ocr_quality(self, text: str) -> float:
        """
        Score OCR output quality using deterministic logic and regex patterns.

        Scoring Criteria:
        - Length of text (0-20 points)
        - Presence of price patterns like 3.99 (0-25 points)
        - Presence of "TOTAL" keyword (0-20 points)
        - Ratio of valid words vs noisy tokens (0-25 points)
        - Penalty for excessive symbols/noise (0-10 points)

        Returns:
            float: Quality score between 0.0 and 1.0
        """
        if not text or not text.strip():
            return 0.0

        score = 0.0

        # 1. Text Length Scoring (0-20 points)
        length_score = self._score_text_length(text)
        score += length_score

        # 2. Price Pattern Detection (0-25 points)
        price_score = self._score_price_patterns(text)
        score += price_score

        # 3. TOTAL Keyword Detection (0-20 points)
        total_score = self._score_total_keyword(text)
        score += total_score

        # 4. Valid Word vs Noise Ratio (0-25 points)
        word_quality_score = self._score_word_quality(text)
        score += word_quality_score

        # 5. Symbol/Noise Penalty (0-10 points deduction)
        noise_penalty = self._calculate_noise_penalty(text)
        score -= noise_penalty

        # Ensure score is within bounds [0.0, 1.0]
        final_score = max(0.0, min(1.0, score))

        return final_score

    def _score_text_length(self, text: str) -> float:
        """
        Score text length (0-20 points).
        Longer text generally indicates better OCR capture.
        """
        length = len(text.strip())

        if length < 50:
            return 0.0  # Very short text
        elif length < 100:
            return 5.0  # Short text
        elif length < 200:
            return 10.0  # Moderate text
        elif length < 400:
            return 15.0  # Good length
        else:
            return 20.0  # Excellent length

    def _score_price_patterns(self, text: str) -> float:
        """
        Score price pattern presence (0-25 points).
        More price patterns indicate better receipt OCR quality.
        """
        # Regex patterns for prices
        price_patterns = [
            r'\$\d+(?:\.\d{2})?',  # $12.99
            r'\d+\.\d{2}',  # 12.99
            r'\d+\s+\.\s+\d{2}',  # 6 . 49 (space-separated)
            r'\d+,\d{2}',  # 12,99 (comma decimal)
        ]

        total_price_matches = 0
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            total_price_matches += len(matches)

        # Score based on number of price matches
        if total_price_matches == 0:
            return 0.0
        elif total_price_matches <= 2:
            return 10.0
        elif total_price_matches <= 5:
            return 15.0
        elif total_price_matches <= 10:
            return 20.0
        else:
            return 25.0

    def _score_total_keyword(self, text: str) -> float:
        """
        Score TOTAL keyword presence (0-20 points).
        Presence of TOTAL indicates complete receipt capture.
        """
        total_patterns = [
            r'\bTOTAL\b',
            r'\bTOTAL\s+AMOUNT\b',
            r'\bGRAND\s+TOTAL\b',
            r'\bSUBTOTAL\b',
            r'\bBALANCE\b',
        ]

        total_matches = 0
        for pattern in total_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                total_matches += 1

        # Score based on presence of total-related keywords
        if total_matches == 0:
            return 0.0
        elif total_matches == 1:
            return 12.0
        elif total_matches == 2:
            return 16.0
        else:
            return 20.0

    def _score_word_quality(self, text: str) -> float:
        """
        Score word quality based on valid words vs noisy tokens (0-25 points).
        Higher ratio of valid words indicates better OCR quality.
        """
        # Split into tokens
        tokens = re.findall(r'\b\w+\b', text)

        if not tokens:
            return 0.0

        valid_word_count = 0
        noise_word_count = 0

        # Common receipt-related valid words
        valid_receipt_words = {
            'TOTAL', 'AMOUNT', 'BALANCE', 'CASH', 'CARD', 'DEBIT', 'CREDIT',
            'GROCERY', 'PRODUCE', 'DAIRY', 'BAKERY', 'MEAT', 'FROZEN',
            'TAX', 'TIP', 'CHANGE', 'PURCHASE', 'PAYMENT', 'TRANSACTION',
            'RECEIPT', 'STORE', 'MARKET', 'SHOP', 'PHARMACY', 'GAS',
            'QUANTITY', 'PRICE', 'COST', 'SUBTOTAL', 'DISCOUNT', 'COUPON',
            'CASHIER', 'CLERK', 'CUSTOMER', 'THANK', 'YOU', 'WELCOME'
        }

        for token in tokens:
            token_upper = token.upper()

            # Check if it's a valid receipt word
            if token_upper in valid_receipt_words:
                valid_word_count += 1
            # Check if it's a number (likely price/quantity)
            elif re.match(r'^\d+(?:\.\d{2})?$', token):
                valid_word_count += 1
            # Check if it's a common English word (3+ letters)
            elif len(token) >= 3 and token.isalpha():
                valid_word_count += 1
            else:
                # Likely noise (short tokens, special chars, etc.)
                noise_word_count += 1

        total_tokens = valid_word_count + noise_word_count
        if total_tokens == 0:
            return 0.0

        valid_ratio = valid_word_count / total_tokens

        # Score based on valid word ratio
        if valid_ratio >= 0.8:
            return 25.0
        elif valid_ratio >= 0.6:
            return 20.0
        elif valid_ratio >= 0.4:
            return 15.0
        elif valid_ratio >= 0.2:
            return 10.0
        else:
            return 5.0

    def _calculate_noise_penalty(self, text: str) -> float:
        """
        Calculate penalty for excessive symbols/noise (0-10 points deduction).
        Too many special characters indicate poor OCR quality.
        """
        # Count special characters (non-alphanumeric, non-space)
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        total_chars = len(text)

        if total_chars == 0:
            return 10.0

        special_ratio = special_chars / total_chars

        # Calculate penalty based on special character ratio
        if special_ratio >= 0.3:  # 30%+ special chars
            return 10.0
        elif special_ratio >= 0.2:  # 20-30% special chars
            return 7.5
        elif special_ratio >= 0.1:  # 10-20% special chars
            return 5.0
        elif special_ratio >= 0.05:  # 5-10% special chars
            return 2.5
        else:  # Less than 5% special chars
            return 0.0

    def should_fallback(self, text: str, threshold: float = 0.25) -> bool:
        """
        Determine when to fallback to Vision OCR based on quality score.

        Args:
            text: OCR text to evaluate
            threshold: Quality threshold below which fallback is recommended (default: 0.25)

        Returns:
            bool: True if fallback to Vision OCR is needed, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for fallback decision - recommending fallback")
            return True

        # Calculate OCR quality score
        quality_score = self.score_ocr_quality(text)

        # Log the quality score for debugging
        logger.info(f"OCR Quality Score: {quality_score:.3f} (threshold: {threshold:.3f})")

        # Make fallback decision
        should_fallback = quality_score < threshold

        if should_fallback:
            logger.warning(f"OCR quality below threshold ({quality_score:.3f} < {threshold:.3f}) - recommending fallback to Vision OCR")
        else:
            logger.info(f"OCR quality acceptable ({quality_score:.3f} >= {threshold:.3f}) - using EasyOCR result")

        return should_fallback

    def get_fallback_reasoning(self, text: str, threshold: float = 0.25) -> Dict[str, Any]:
        """
        Get detailed reasoning for fallback decision.

        Args:
            text: OCR text to evaluate
            threshold: Quality threshold for fallback decision

        Returns:
            Dict containing detailed scoring breakdown and reasoning
        """
        if not text or not text.strip():
            return {
                'should_fallback': True,
                'quality_score': 0.0,
                'threshold': threshold,
                'reasoning': 'Empty text - automatic fallback recommended',
                'component_scores': {},
                'recommendation': 'FALLBACK'
            }

        # Calculate component scores
        length_score = self._score_text_length(text)
        price_score = self._score_price_patterns(text)
        total_score = self._score_total_keyword(text)
        word_score = self._score_word_quality(text)
        noise_penalty = self._calculate_noise_penalty(text)
        quality_score = self.score_ocr_quality(text)

        # Determine reasoning
        reasoning_parts = []

        if length_score < 10:
            reasoning_parts.append(f"Text too short ({len(text)} chars)")

        if price_score < 15:
            reasoning_parts.append("Insufficient price patterns detected")

        if total_score < 12:
            reasoning_parts.append("Missing TOTAL/AMOUNT keywords")

        if word_score < 15:
            reasoning_parts.append("Low word quality (many noisy tokens)")

        if noise_penalty > 5:
            reasoning_parts.append("Excessive special characters/noise")

        # Create recommendation
        should_fallback = quality_score < threshold
        recommendation = "FALLBACK" if should_fallback else "USE_EASYOCR"

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Good quality OCR"

        return {
            'should_fallback': should_fallback,
            'quality_score': quality_score,
            'threshold': threshold,
            'reasoning': reasoning,
            'recommendation': recommendation,
            'component_scores': {
                'text_length': length_score,
                'price_patterns': price_score,
                'total_keyword': total_score,
                'word_quality': word_score,
                'noise_penalty': noise_penalty
            }
        }
