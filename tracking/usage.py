"""Token usage tracking and cost estimation utilities"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ModelPricing(Enum):
    """Pricing information for different models"""
    GPT_4O_MINI = {
        "input_cost_per_1k": 0.15,    # $0.15 per 1K input tokens
        "output_cost_per_1k": 0.60,   # $0.60 per 1K output tokens
        "name": "gpt-4o-mini"
    }
    GPT_4O = {
        "input_cost_per_1k": 2.50,    # $2.50 per 1K input tokens
        "output_cost_per_1k": 10.00,   # $10.00 per 1K output tokens
        "name": "gpt-4o"
    }
    GPT_3_5_TURBO = {
        "input_cost_per_1k": 0.50,    # $0.50 per 1K input tokens
        "output_cost_per_1k": 1.50,    # $1.50 per 1K output tokens
        "name": "gpt-3.5-turbo"
    }


@dataclass
class TokenUsage:
    """Tracks token usage and cost estimation"""
    input_tokens: int = 0
    output_tokens: int = 0
    model_name: str = "gpt-4o-mini"

    def add_usage(self, input_tokens: int, output_tokens: int, model: Optional[str] = None):
        """Add token usage from a single operation"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if model:
            self.model_name = model
        logger.debug(f"Added usage: +{input_tokens} input, +{output_tokens} output tokens")

    def get_total_tokens(self) -> int:
        """Get total tokens used"""
        return self.input_tokens + self.output_tokens

    def get_estimated_cost(self, model: Optional[Union[str, ModelPricing]] = None) -> float:
        """Estimate cost based on token usage and model"""
        if model is None:
            model = self.model_name
        
        # Get pricing information
        if isinstance(model, str):
            try:
                pricing = ModelPricing[model.upper().replace('-', '_')]
                pricing_info = pricing.value
            except KeyError:
                # Default to gpt-4o-mini pricing if model not found
                logger.warning(f"Unknown model {model}, using gpt-4o-mini pricing")
                pricing_info = ModelPricing.GPT_4O_MINI.value
        else:
            pricing_info = model.value

        input_cost_per_1k = pricing_info["input_cost_per_1k"]
        output_cost_per_1k = pricing_info["output_cost_per_1k"]

        input_cost = (self.input_tokens / 1000) * input_cost_per_1k
        output_cost = (self.output_tokens / 1000) * output_cost_per_1k

        total_cost = input_cost + output_cost
        logger.debug(f"Estimated cost for {pricing_info['name']}: ${total_cost:.4f}")
        return total_cost

    def get_summary(self, model: Optional[Union[str, ModelPricing]] = None) -> str:
        """Get formatted summary of token usage"""
        cost = self.get_estimated_cost(model)
        model_name = self._get_model_display_name(model)
        
        return (f"Token Usage Summary:\n"
                f"Model: {model_name}\n"
                f"Input Tokens: {self.input_tokens:,}\n"
                f"Output Tokens: {self.output_tokens:,}\n"
                f"Total Tokens: {self.get_total_tokens():,}\n"
                f"Estimated Cost: ${cost:.4f}")

    def reset(self):
        """Reset token usage counters"""
        self.input_tokens = 0
        self.output_tokens = 0
        logger.info("Token usage reset")

    def to_dict(self, model: Optional[Union[str, ModelPricing]] = None) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.get_total_tokens(),
            "estimated_cost": self.get_estimated_cost(model),
            "model_name": self.model_name
        }

    def _get_model_display_name(self, model: Optional[Union[str, ModelPricing]] = None) -> str:
        """Get display name for model"""
        if model is None:
            model = self.model_name
        
        if isinstance(model, str):
            return model
        else:
            return model.value["name"]


class TokenTracker:
    """Advanced token tracking with multiple operations and cost analysis"""
    
    def __init__(self, default_model: str = "gpt-4o-mini"):
        self.operations: list[Dict[str, Any]] = []
        self.default_model = default_model
        self.total_usage = TokenUsage(model_name=default_model)
    
    def track_operation(self, 
                      operation_name: str,
                      input_tokens: int,
                      output_tokens: int,
                      model: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """Track a single operation's token usage"""
        if model is None:
            model = self.default_model
        
        operation = {
            "name": operation_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model,
            "timestamp": self._get_timestamp(),
            "cost": self._calculate_operation_cost(input_tokens, output_tokens, model),
            "metadata": metadata or {}
        }
        
        self.operations.append(operation)
        self.total_usage.add_usage(input_tokens, output_tokens, model)
        
        logger.info(f"Tracked operation '{operation_name}': {input_tokens} input, {output_tokens} output tokens")
    
    def get_operation_summary(self) -> str:
        """Get summary of all operations"""
        if not self.operations:
            return "No operations tracked"
        
        summary_lines = [
            f"Token Tracking Summary ({len(self.operations)} operations):",
            f"Total Input: {self.total_usage.input_tokens:,} tokens",
            f"Total Output: {self.total_usage.output_tokens:,} tokens", 
            f"Total Tokens: {self.total_usage.get_total_tokens():,}",
            f"Total Cost: ${self.total_usage.get_estimated_cost():.4f}",
            "",
            "Operations:"
        ]
        
        for i, op in enumerate(self.operations, 1):
            summary_lines.append(
                f"  {i}. {op['name']}: {op['input_tokens']}→{op['output_tokens']} tokens, "
                f"${op['cost']:.4f} ({op['model']})"
            )
        
        return "\n".join(summary_lines)
    
    def get_most_expensive_operation(self, limit: int = 5) -> list[Dict[str, Any]]:
        """Get the most expensive operations"""
        sorted_ops = sorted(self.operations, key=lambda x: x['cost'], reverse=True)
        return sorted_ops[:limit]
    
    def get_usage_by_model(self) -> Dict[str, TokenUsage]:
        """Get token usage breakdown by model"""
        usage_by_model = {}
        
        for op in self.operations:
            model = op['model']
            if model not in usage_by_model:
                usage_by_model[model] = TokenUsage(model_name=model)
            
            usage_by_model[model].add_usage(op['input_tokens'], op['output_tokens'])
        
        return usage_by_model
    
    def reset(self):
        """Reset all tracking data"""
        self.operations.clear()
        self.total_usage.reset()
        logger.info("Token tracker reset")
    
    def _calculate_operation_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for a single operation"""
        temp_usage = TokenUsage(model_name=model)
        temp_usage.add_usage(input_tokens, output_tokens)
        return temp_usage.get_estimated_cost()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


def extract_token_usage(response) -> tuple[int, int, int]:
    """Extract token usage from OpenAI response"""
    usage_data = getattr(response, 'usage_metadata', {})
    input_tokens = usage_data.get('input_tokens', 0)
    output_tokens = usage_data.get('output_tokens', 0)
    return input_tokens, output_tokens, input_tokens + output_tokens


def extract_token_usage_from_response(response) -> Dict[str, int]:
    """Extract token usage from response as dictionary"""
    usage_data = getattr(response, 'usage_metadata', {})
    return {
        'input_tokens': usage_data.get('input_tokens', 0),
        'output_tokens': usage_data.get('output_tokens', 0),
        'total_tokens': usage_data.get('input_tokens', 0) + usage_data.get('output_tokens', 0)
    }


def calculate_cost_estimate(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
    """Calculate cost estimate for given token usage"""
    usage = TokenUsage()
    usage.add_usage(input_tokens, output_tokens, model)
    return usage.get_estimated_cost()


def format_cost_summary(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> str:
    """Format cost summary for display"""
    usage = TokenUsage()
    usage.add_usage(input_tokens, output_tokens, model)
    return usage.get_summary()
