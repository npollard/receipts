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

        # Handle both dict and ReceiptModel data
        if hasattr(result.data, 'model_dump'):
            # It's a Pydantic model
            data_dict = result.data.model_dump()
        else:
            # It's already a dict
            data_dict = result.data or {}

        # Look for token usage in different possible locations
        token_usage_data = None
        if '_token_usage' in data_dict:
            token_usage_data = data_dict['_token_usage']
        elif 'parsed_receipt' in data_dict and '_token_usage' in data_dict['parsed_receipt']:
            token_usage_data = data_dict['parsed_receipt']['_token_usage']

        return token_usage_data

    def aggregate_token_usage(self, results: list[APIResponse]) -> TokenUsage:
        """Aggregate token usage from multiple results"""
        total_token_usage = TokenUsage()

        for result in results:
            if result.status == 'success' and result.data:
                token_usage_data = self.extract_token_usage_from_result(result)
                if token_usage_data:
                    total_token_usage.add_usage(
                        token_usage_data.get('input_tokens', 0),
                        token_usage_data.get('output_tokens', 0)
                    )

        return total_token_usage

    def save_token_usage_to_persistence(self, token_usage: TokenUsage) -> Optional[str]:
        """Save token usage to persistent storage"""
        if token_usage.get_total_tokens() > 0:
            session_id = f"batch_session_{token_usage.get_total_tokens()}"
            self.persistence.save_usage(token_usage, session_id)
            self.logger.info(f"Saved batch token usage to persistent storage: {session_id}")
            return session_id
        return None

    def print_token_usage_summary(self, token_usage: TokenUsage):
        """Print token usage summary for current batch"""
        self.logger.info("=" * 50)
        self.logger.info("TOKEN USAGE SUMMARY")
        self.logger.info(token_usage.get_summary())
        self.logger.info("=" * 50)

    def print_usage_summary(self, show_persisted: bool = False):
        """Print token usage summary from persistent storage"""
        summary = self.persistence.get_usage_summary()

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
