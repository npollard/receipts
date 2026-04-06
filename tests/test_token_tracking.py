"""Unit tests for TokenUsage"""

import pytest
from token_tracking import TokenUsage


def test_token_usage_initialization():
    """Test TokenUsage initialization"""
    token_usage = TokenUsage()

    assert token_usage.input_tokens == 0
    assert token_usage.output_tokens == 0
    assert token_usage.get_total_tokens() == 0


def test_token_usage_add_usage():
    """Test adding token usage"""
    token_usage = TokenUsage()
    token_usage.add_usage(100, 50)

    assert token_usage.input_tokens == 100
    assert token_usage.output_tokens == 50
    assert token_usage.get_total_tokens() == 150


def test_token_usage_add_multiple_uses():
    """Test adding multiple token usage entries"""
    token_usage = TokenUsage()
    token_usage.add_usage(100, 50)
    token_usage.add_usage(200, 75)

    assert token_usage.input_tokens == 300
    assert token_usage.output_tokens == 125
    assert token_usage.get_total_tokens() == 425


def test_token_usage_get_total_tokens():
    """Test getting total tokens"""
    token_usage = TokenUsage()
    token_usage.add_usage(150, 75)

    assert token_usage.get_total_tokens() == 225


def test_token_usage_get_summary():
    """Test getting usage summary"""
    token_usage = TokenUsage()
    token_usage.add_usage(1000, 500)

    summary = token_usage.get_summary()

    assert "Input Tokens: 1,000" in summary
    assert "Output Tokens: 500" in summary
    assert "Total Tokens: 1,500" in summary
    assert "Estimated Cost: $" in summary


def test_token_usage_reset():
    """Test resetting token usage"""
    token_usage = TokenUsage()
    token_usage.add_usage(100, 50)
    token_usage.reset()

    assert token_usage.input_tokens == 0
    assert token_usage.output_tokens == 0
    assert token_usage.get_total_tokens() == 0


def test_token_usage_to_dict():
    """Test TokenUsage to_dict conversion"""
    token_usage = TokenUsage()
    token_usage.add_usage(100, 50)

    data = token_usage.to_dict()

    assert data["input_tokens"] == 100
    assert data["output_tokens"] == 50
    assert data["total_tokens"] == 150
    assert "estimated_cost" in data


def test_token_usage_get_estimated_cost():
    """Test getting estimated cost"""
    token_usage = TokenUsage()
    token_usage.add_usage(1000, 500)  # 1K input, 0.5K output

    cost = token_usage.get_estimated_cost()

    # Expected: (1000/1000 * 0.15) + (500/1000 * 0.60) = 0.15 + 0.30 = 0.45
    assert abs(cost - 0.45) < 0.001
