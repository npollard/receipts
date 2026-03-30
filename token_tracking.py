"""Token usage tracking utilities"""

import logging
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Tracks token usage and cost estimation"""
    input_tokens: int = 0
    output_tokens: int = 0

    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage from a single operation"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        logger.debug(f"Added usage: +{input_tokens} input, +{output_tokens} output tokens")

    def get_total_tokens(self) -> int:
        """Get total tokens used"""
        return self.input_tokens + self.output_tokens

    def get_estimated_cost(self) -> float:
        """Estimate cost based on token usage"""
        # GPT-4o-mini pricing (current as of 2024)
        input_cost_per_1k = 0.15  # $0.15 per 1K input tokens
        output_cost_per_1k = 0.60  # $0.60 per 1K output tokens

        input_cost = (self.input_tokens / 1000) * input_cost_per_1k
        output_cost = (self.output_tokens / 1000) * output_cost_per_1k

        total_cost = input_cost + output_cost
        logger.debug(f"Estimated cost: ${total_cost:.4f}")
        return total_cost

    def get_summary(self) -> str:
        """Get formatted summary of token usage"""
        return (f"Token Usage Summary:\n"
                f"Input Tokens: {self.input_tokens:,}\n"
                f"Output Tokens: {self.output_tokens:,}\n"
                f"Total Tokens: {self.get_total_tokens():,}\n"
                f"Estimated Cost: ${self.get_estimated_cost():.4f}")

    def reset(self):
        """Reset token usage counters"""
        self.input_tokens = 0
        self.output_tokens = 0
        logger.info("Token usage reset")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.get_total_tokens(),
            "estimated_cost": self.get_estimated_cost()
        }
