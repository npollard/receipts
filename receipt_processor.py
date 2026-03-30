"""Main receipt processor class"""

from image_processing import VisionProcessor
from ai_parsing import ReceiptParser
from workflow import WorkflowOrchestrator
from token_tracking import TokenUsage
from api_response import APIResponse
from typing import Dict, Any


class ReceiptProcessor:
    """Main processor class that coordinates all components"""

    def __init__(self):
        self.image_processor = VisionProcessor()
        self.ai_parser = ReceiptParser()
        self.orchestrator = WorkflowOrchestrator(self.image_processor, self.ai_parser)


    def process_directly(self, image_path: str) -> APIResponse:
        """Process directly without LangGraph"""
        return self.orchestrator.process_image(image_path)

    def get_token_usage_summary(self) -> str:
        """Get current token usage summary"""
        return self.orchestrator.get_token_usage_summary()

    def reset_token_usage(self):
        """Reset token usage tracking"""
        self.orchestrator.token_usage = TokenUsage()
