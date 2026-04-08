"""User management for receipt processing"""

import logging
import os
from typing import Optional
from uuid import UUID

from database_models import DatabaseManager, User
from storage import UserRepository
from receipt_persistence import ReceiptPersistence

from config import DEFAULT_USER_EMAIL

logger = logging.getLogger(__name__)

class UserManager:
    """Manages user context for receipt processing"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.user_repository = UserRepository(db_manager.engine.url)
        self._current_user_id = None
        self._current_user = None

    def get_or_create_default_user(self) -> User:
        """Get or create a default user for single-user mode"""
        # Use config default email
        default_email = DEFAULT_USER_EMAIL

        user = self.user_repository.get_or_create_user(default_email)
        self._current_user = user
        self._current_user_id = user.id
        logger.info(f"Using default user: {default_email} ({user.id})")
        return user

    def get_current_user_id(self) -> UUID:
        """Get the current user's ID"""
        if self._current_user_id is None:
            self.get_or_create_default_user()
        return self._current_user_id

    def get_current_user(self) -> User:
        """Get the current user object"""
        if self._current_user is None:
            self.get_or_create_default_user()
        return self._current_user

    def set_user_by_email(self, email: str) -> User:
        """Set current user by email (creates if not found)"""
        user = self.user_repository.get_or_create_user(email)
        self._current_user = user
        self._current_user_id = user.id
        return user

    def set_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Set current user by ID (returns None if not found)"""
        user = self.user_repository.get_user_by_id(user_id)
        if user:
            self._current_user = user
            self._current_user_id = user_id
        return user

    def is_multi_user_mode(self) -> bool:
        """Check if system is configured for multi-user mode"""
        return os.getenv("MULTI_USER_MODE", "false").lower() == "true"

    def get_user_context(self) -> dict:
        """Get user context information"""
        user = self.get_current_user()
        return {
            "user_id": str(user.id),
            "email": user.email,
            "is_multi_user": self.is_multi_user_mode(),
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
