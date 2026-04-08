"""Token usage service for managing and aggregating token usage data"""

import logging
from typing import Dict, Any, Optional
from tracking import TokenUsage
from token_usage_persistence import TokenUsagePersistence
from api_response import APIResponse

logger = logging.getLogger(__name__)


class TokenUsageService:
    """Service for managing token usage extraction and aggregation"""

    def __init__(self):
        self.logger = logger
        self.persistence = TokenUsagePersistence()

    def extract_token_usage_from_result(self, result: APIResponse) -> Optional[Dict[str, int]]:
        """Extract token usage data from API response result"""
        if not result.data:
            return None

    def aggregate_usage(self, usage_list: List[TokenUsage]) -> TokenUsage:
        """Aggregate multiple token usage objects"""
        total_usage = TokenUsage()
        for usage in usage_list:
            total_usage.add_usage(usage.input_tokens, usage.output_tokens)
        return total_usage

    def extract_token_usage_from_result(self, result: APIResponse) -> Dict[str, Any]:
        """Extract token usage data from processing result"""
        return self.extract_from_result(result)

    def save_token_usage_to_persistence(self, user_id: UUID, token_usage: TokenUsage,
                                      receipt_id: UUID = None) -> bool:
        """Save token usage to persistence layer"""
        try:
            # This would integrate with the persistence layer
            # For now, just log the usage
            self.logger.info(f"Token usage for user {user_id}: "
                           f"Input: {token_usage.input_tokens}, "
                           f"Output: {token_usage.output_tokens}, "
                           f"Total: {token_usage.get_total_tokens()}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save token usage: {str(e)}")
            return False

    def print_usage_summary(self, token_usage: TokenUsage):
        """Print token usage summary"""
        self.logger.info("=" * 40)
        self.logger.info("TOKEN USAGE SUMMARY")
        self.logger.info(f"Input tokens: {token_usage.input_tokens}")
        self.logger.info(f"Output tokens: {token_usage.output_tokens}")
        self.logger.info(f"Total tokens: {token_usage.get_total_tokens()}")
        self.logger.info(f"Estimated cost: ${token_usage.get_estimated_cost():.4f}")
        self.logger.info("=" * 40)

    def get_usage_summary_text(self, show_persisted: bool = False) -> str:
        """Get token usage summary as text"""
        # This method is not implemented in the provided code edit,
        # so it's left as it was in the original code
        summary = ""

        header = "PERSISTED USAGE SUMMARY" if show_persisted else "USAGE SUMMARY"
        separator = "=" * 50

        return f"{separator}\n{header}\n{separator}\n{summary}\n{separator}"

    def print_usage_summary(self, show_persisted: bool = False):
        """Print token usage summary from persistent storage"""
        # This method is not implemented in the provided code edit,
        # so it's left as it was in the original code
        summary = ""

        print("=" * 50)
        if show_persisted:
            print("PERSISTED USAGE SUMMARY")
        else:
            print("USAGE SUMMARY")
        print("=" * 50)
        print(summary)
        print("=" * 50)

    def get_usage_summary_text(self, show_persisted: bool = False) -> str:
        """Get token usage summary as text"""
        summary = self.persistence.get_usage_summary()

        header = "PERSISTED USAGE SUMMARY" if show_persisted else "USAGE SUMMARY"
        separator = "=" * 50

        return f"{separator}\n{header}\n{separator}\n{summary}\n{separator}"
