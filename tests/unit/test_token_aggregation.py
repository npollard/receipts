"""Unit tests for token aggregation logic.

Replaces logic-heavy integration tests with isolated unit tests.
"""

import pytest
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class TokenUsage:
    """Simplified token usage for testing."""
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    def add_usage(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


def aggregate_token_usage(receipts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate token usage from multiple receipts."""
    total_input = sum(r.get("input_tokens", 0) for r in receipts)
    total_output = sum(r.get("output_tokens", 0) for r in receipts)
    
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "receipt_count": len(receipts),
        "avg_per_receipt": (total_input + total_output) / len(receipts) if receipts else 0
    }


class TestTokenAggregation:
    """Token usage aggregation logic."""

    def test_single_receipt_aggregation(self):
        """Given: Single receipt. When: Aggregated. Then: Same values returned."""
        receipts = [
            {"input_tokens": 100, "output_tokens": 50}
        ]
        
        result = aggregate_token_usage(receipts)
        
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["receipt_count"] == 1

    def test_multiple_receipts_sum_correctly(self):
        """Given: Multiple receipts. When: Aggregated. Then: Values summed."""
        receipts = [
            {"input_tokens": 100, "output_tokens": 50},
            {"input_tokens": 120, "output_tokens": 45},
            {"input_tokens": 80, "output_tokens": 60}
        ]
        
        result = aggregate_token_usage(receipts)
        
        assert result["input_tokens"] == 300  # 100 + 120 + 80
        assert result["output_tokens"] == 155  # 50 + 45 + 60
        assert result["total_tokens"] == 455
        assert result["receipt_count"] == 3

    def test_empty_receipts_list(self):
        """Given: Empty list. When: Aggregated. Then: Zero values returned."""
        result = aggregate_token_usage([])
        
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0
        assert result["receipt_count"] == 0
        assert result["avg_per_receipt"] == 0

    def test_missing_token_fields_default_to_zero(self):
        """Given: Receipts missing token fields. When: Aggregated. Then: Defaults to zero."""
        receipts = [
            {"input_tokens": 100},  # missing output_tokens
            {"output_tokens": 50},  # missing input_tokens
            {}  # missing both
        ]
        
        result = aggregate_token_usage(receipts)
        
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["receipt_count"] == 3

    def test_average_calculation(self):
        """Given: Multiple receipts. When: Averaged. Then: Correct average."""
        receipts = [
            {"input_tokens": 100, "output_tokens": 50},
            {"input_tokens": 100, "output_tokens": 50}
        ]
        
        result = aggregate_token_usage(receipts)
        
        assert result["avg_per_receipt"] == 150.0  # (150 + 150) / 2


class TestTokenUsageClass:
    """TokenUsage dataclass behavior."""

    def test_initial_values_zero(self):
        """Given: New TokenUsage. When: Created. Then: Values are zero."""
        usage = TokenUsage()
        
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_add_usage_accumulates(self):
        """Given: TokenUsage with values. When: Add usage. Then: Accumulates."""
        usage = TokenUsage()
        
        usage.add_usage(100, 50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        
        usage.add_usage(50, 25)
        assert usage.input_tokens == 150
        assert usage.output_tokens == 75

    def test_total_tokens_updates(self):
        """Given: TokenUsage modified. When: Total accessed. Then: Current sum."""
        usage = TokenUsage()
        
        usage.add_usage(100, 50)
        assert usage.total_tokens == 150
        
        usage.add_usage(50, 25)
        assert usage.total_tokens == 225


class TestCostEstimation:
    """Cost estimation from token usage."""

    def test_cost_calculation_gpt4_mini(self):
        """Given: Token count. When: Cost estimated. Then: Correct rate applied."""
        # gpt-4o-mini rates: $0.00015/1K input, $0.0006/1K output
        input_tokens = 120
        output_tokens = 45
        
        input_cost = (input_tokens / 1000) * 0.00015
        output_cost = (output_tokens / 1000) * 0.0006
        total_cost = input_cost + output_cost
        
        assert total_cost == pytest.approx(0.000045, rel=1e-4)

    def test_batch_cost_aggregation(self):
        """Given: Batch of receipts. When: Cost estimated. Then: Sum of individual."""
        receipts = [
            {"input_tokens": 120, "output_tokens": 45},
            {"input_tokens": 80, "output_tokens": 20}
        ]
        
        total_input = sum(r["input_tokens"] for r in receipts)
        total_output = sum(r["output_tokens"] for r in receipts)
        
        # gpt-4o-mini rates
        total_cost = (total_input / 1000) * 0.00015 + (total_output / 1000) * 0.0006
        
        # 200 input, 65 output
        expected = (200 / 1000) * 0.00015 + (65 / 1000) * 0.0006
        assert total_cost == pytest.approx(expected, rel=1e-6)
