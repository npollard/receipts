"""Main receipt processor class"""

from image_processing import VisionProcessor
from ai_parsing import ReceiptParser
from workflow import WorkflowOrchestrator
from token_tracking import TokenUsage
from api_response import APIResponse
from database_models import DatabaseManager
from typing import Dict, Any, Optional
from uuid import UUID


class ReceiptProcessor:
    """Main processor class that coordinates all components"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None, user_id: Optional[UUID] = None):
        self.image_processor = VisionProcessor()
        self.ai_parser = ReceiptParser()
        self.orchestrator = WorkflowOrchestrator(
            self.image_processor,
            self.ai_parser,
            db_manager=db_manager,
            user_id=user_id
        )

    def process_directly(self, image_path: str) -> APIResponse:
        """Process directly without LangGraph"""
        return self.orchestrator.process_image(image_path)

    def get_token_usage_summary(self) -> str:
        """Get current token usage summary"""
        return self.orchestrator.get_token_usage_summary()

    def reset_token_usage(self):
        """Reset token usage tracking"""
        self.orchestrator.token_usage = TokenUsage()

    def get_current_user(self):
        """Get the current user object"""
        return self.orchestrator.get_current_user()

    def get_current_user_id(self) -> Optional[UUID]:
        """Get the current user's ID"""
        return self.orchestrator.get_current_user_id()

    def switch_user(self, email: str):
        """Switch to a different user (creates if doesn't exist)"""
        return self.orchestrator.switch_user(email)

    def get_user_context(self) -> dict:
        """Get current user context information"""
        return self.orchestrator.get_user_context()

    def get_user_receipts(self, limit: int = 50, offset: int = 0, status: Optional[str] = None):
        """Get user's receipts from database (if persistence enabled)"""
        return self.orchestrator.get_user_receipts(limit, offset, status)
