"""Token usage utilities for OpenAI responses"""

import logging

logger = logging.getLogger(__name__)


def extract_token_usage(response) -> tuple[int, int, int]:
    """Extract token usage from OpenAI response"""
    usage_data = getattr(response, 'usage_metadata', {})
    input_tokens = usage_data.get('input_tokens', 0)
    output_tokens = usage_data.get('output_tokens', 0)
    return input_tokens, output_tokens, input_tokens + output_tokens
