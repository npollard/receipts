"""OCR quality scoring for receipt text

Business rules for evaluating OCR output quality based on
receipt-specific content patterns.
"""

import re
from typing import Dict, Any, Optional


def score_ocr_quality(text: str, detailed: bool = False, debug: bool = False) -> float:
    """Score OCR output quality using deterministic logic and regex patterns.

    Scoring Criteria:
    - Length of text (0-20 points)
    - Presence of price patterns like 3.99 (0-25 points)
    - Presence of "TOTAL" keyword (0-20 points)
    - Ratio of valid words vs noisy tokens (0-25 points)
    - Penalty for excessive symbols/noise (0-10 points)

    Args:
        text: OCR text to evaluate
        detailed: If True, return dict with component scores instead of just total score
        debug: If True, print detailed scoring breakdown (via return value)

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

    # Precompute shared values to avoid redundant operations
    text_stripped = text.strip()
    words_lower = set(re.findall(r"\b\w+\b", text.lower()))
    price_matches = re.findall(r'\$?\d+\.\d{2}', text)

    # Calculate component scores (passing precomputed values)
    length_score = _score_text_length(text_stripped)
    price_score = _score_price_patterns(text, price_matches)
    total_score = _score_total_keyword(words_lower)
    word_quality_score = _score_word_quality(text)
    noise_penalty = _calculate_noise_penalty(text, text_stripped)

    # Calculate total score
    raw_total = length_score + price_score + total_score + word_quality_score - noise_penalty
    final_score = max(0.0, min(1.0, raw_total / 100.0))  # Normalize to 0-1

    # Final guard: receipts must have prices OR total keywords
    has_price = len(price_matches) > 0
    has_total = any(kw in words_lower for kw in ["total", "subtotal", "amount"])

    if not has_price and not has_total:
        final_score = min(final_score, 0.4)

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


def _score_text_length(text_stripped: str) -> float:
    """Score text length (0-20 points)"""
    return min(len(text_stripped) / 10, 20)


def _score_price_patterns(text: str, precomputed_matches: list = None) -> float:
    """Score price patterns (0-25 points).

    Args:
        text: The OCR text to score
        precomputed_matches: Optional precomputed price matches to avoid redundant regex
    """
    # Use precomputed matches if available, otherwise compute
    if precomputed_matches is not None:
        price_matches = len(precomputed_matches)
    else:
        price_matches = len(re.findall(r'\$?\d+\.\d{2}', text))

    # Match space-separated prices like "3 99" but not "123 456" sequences
    space_prices = len(re.findall(r'(?<!\d)\d{1,2}\s+\d{2}(?!\d)', text))
    total_matches = price_matches + (space_prices * 0.5)
    return min(total_matches * 5, 25)


def _score_total_keyword(words_lower: set) -> float:
    """Score total keywords (0-20 points).

    Args:
        words_lower: Set of lowercase words (precomputed to avoid redundant regex)
    """
    total_keywords = {"total", "subtotal", "amount"}
    total_count = sum(1 for kw in total_keywords if kw in words_lower)
    # Boost weight: 7 points per keyword to ensure >= 0.6 with 2 keywords
    return min(total_count * 7, 20)


def _score_word_quality(text: str) -> float:
    """Score word quality based on receipt-relevant content (0-25 points)"""
    words = text.split()
    if not words:
        return 0

    valid_count = 0
    for word in words:
        # Valid if: has letters, OR is a price pattern (e.g., 3.99, $5.00)
        # OR is a reasonable numeric value (not just noise)
        clean = word.strip('$,.:;')
        has_letters = any(c.isalpha() for c in word)
        is_price = bool(re.match(r'^\$?\d+\.\d{2}$', clean))
        is_number = clean.isdigit() and len(clean) <= 6  # Reasonable numeric values

        if has_letters or is_price or is_number:
            valid_count += 1

    return (valid_count / len(words) * 25)


def _calculate_noise_penalty(text: str, text_stripped: str = None) -> float:
    """Calculate noise penalty (0-10 points).

    Args:
        text: The OCR text to score
        text_stripped: Optional pre-stripped text to avoid redundant strip()
    """
    noise_chars = len(re.findall(r'[^\w\s$.,%\-:\/\n]', text))
    text_len = len(text_stripped) if text_stripped is not None else len(text.strip())
    noise_ratio = noise_chars / text_len if text_len > 0 else 0
    if noise_ratio > 0.3:
        return 4.0
    return 0.0


def should_fallback(text: str, threshold: float) -> bool:
    """Determine if OCR quality is below threshold and fallback is needed"""
    quality_score = score_ocr_quality(text)
    return quality_score < threshold


def get_fallback_reasoning(text: str, threshold: float) -> Dict[str, Any]:
    """Get detailed reasoning for fallback decision"""
    quality_score = score_ocr_quality(text, detailed=True)

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
