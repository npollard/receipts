"""User management utilities for receipt processing"""

import os
from typing import Optional
from uuid import UUID, uuid4
from dotenv import load_dotenv

from database_models import DatabaseManager, User
from receipt_persistence import ReceiptPersistence

load_dotenv()

class UserManager:
    """Manages user context for receipt processing"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._current_user_id = None
        self._current_user = None
    
    def get_or_create_default_user(self) -> User:
        """Get or create a default user for single-user mode"""
        # Use environment variable or create a default
        default_email = os.getenv("DEFAULT_USER_EMAIL", "user@receipts.local")
        
        persistence = ReceiptPersistence(self.db_manager, uuid4())  # Temp user ID for lookup
        user = persistence.get_or_create_user(default_email)
        
        self._current_user_id = user.id
        self._current_user = user
        
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
        """Set current user by email (creates if doesn't exist)"""
        persistence = ReceiptPersistence(self.db_manager, self.get_current_user_id())
        user = persistence.get_or_create_user(email)
        
        self._current_user_id = user.id
        self._current_user = user
        
        return user
    
    def set_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Set current user by ID (returns None if not found)"""
        session = self.db_manager.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                self._current_user_id = user.id
                self._current_user = user
            return user
        except Exception as e:
            raise e
        finally:
            session.close()
    
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
