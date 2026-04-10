"""Token usage service for managing and aggregating token usage data"""

import logging
from typing import Dict, Any, Optional, List
from uuid import UUID
from tracking import TokenUsage
from api_response import APIResponse

logger = logging.getLogger(__name__)


class TokenUsageService:
    """Service for managing token usage extraction and aggregation"""

    def __init__(self):
        self.logger = logger

    def extract_token_usage_from_result(self, result: APIResponse) -> Optional[Dict[str, int]]:
        """Extract token usage data from API response result"""
        if not result.data:
            return None
        # TODO: Implement token usage extraction from result
        return None

    def aggregate_usage(self, usage_list: List[TokenUsage]) -> TokenUsage:
        """Aggregate multiple token usage objects"""
        total_usage = TokenUsage()
        for usage in usage_list:
            total_usage.add_usage(usage.input_tokens, usage.output_tokens)
        return total_usage

    def save_token_usage_to_persistence(self, user_id: UUID, token_usage: TokenUsage,
                                      receipt_id: UUID = None) -> bool:
        """Save token usage to persistence layer (stored in Receipt table)"""
        # Token usage is now persisted via ReceiptRepository.save_receipt()
        # This method is kept for backward compatibility but tokens are stored
        # as part of the receipt record in the database
        self.logger.debug(f"Token usage for receipt {receipt_id}: "
                         f"Input: {token_usage.input_tokens}, "
                         f"Output: {token_usage.output_tokens}, "
                         f"Total: {token_usage.get_total_tokens()}")
        return True

    def print_usage_summary(self, show_persisted: bool = False):
        """Print token usage summary"""
        summary = self.get_usage_summary()

        print("=" * 50)
        print("USAGE SUMMARY")
        print("=" * 50)
        print(f"Total Sessions: {summary.get('total_sessions', 0)}")
        print(f"Total Input Tokens: {summary.get('total_input_tokens', 0)}")
        print(f"Total Output Tokens: {summary.get('total_output_tokens', 0)}")
        print(f"Total Tokens: {summary.get('total_tokens', 0)}")
        print(f"Total Estimated Cost: ${summary.get('total_estimated_cost', 0):.4f}")
        print("=" * 50)

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary - aggregated from receipt records"""
        # Note: For full DB integration, this should query ReceiptRepository
        # For now, return empty summary (actual aggregation done via DB queries)
        return {
            "total_sessions": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_estimated_cost": 0.0
        }

    def get_usage_summary_text(self, show_persisted: bool = False) -> str:
        """Get token usage summary as text"""
        summary = self.get_usage_summary()

        summary_text = (
            f"Total Sessions: {summary.get('total_sessions', 0)}\n"
            f"Total Input Tokens: {summary.get('total_input_tokens', 0)}\n"
            f"Total Output Tokens: {summary.get('total_output_tokens', 0)}\n"
            f"Total Tokens: {summary.get('total_tokens', 0)}\n"
            f"Total Estimated Cost: ${summary.get('total_estimated_cost', 0):.4f}"
        )

        header = "USAGE SUMMARY"
        separator = "=" * 50

        return f"{separator}\n{header}\n{separator}\n{summary_text}\n{separator}"
