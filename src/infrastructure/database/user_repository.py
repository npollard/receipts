"""User repository for database operations using unified session management"""

import logging
from typing import Optional
from uuid import UUID

from .session import get_session, get_read_session
from .repository import handle_uuid_for_db
from .models import User

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for user database operations"""

    def __init__(self, database_url: Optional[str] = None):
        # database_url kept for backward compatibility but ignored
        pass

    def get_user_by_email(self, email: str) -> Optional[dict]:
        """Get user by email"""
        with get_read_session() as session:
            user = session.query(User).filter(User.email == email).first()
            if user:
                return {
                    "id": str(user.id) if user.id else None,
                    "email": user.email,
                    "created_at": user.created_at,
                    "updated_at": user.updated_at,
                    "is_active": user.is_active,
                }
            return None

    def get_user_by_id(self, user_id: UUID) -> Optional[dict]:
        """Get user by ID"""
        user_id_for_db = handle_uuid_for_db(user_id)
        with get_read_session() as session:
            user = session.query(User).filter(User.id == user_id_for_db).first()
            if user:
                return {
                    "id": str(user.id) if user.id else None,
                    "email": user.email,
                    "created_at": user.created_at,
                    "updated_at": user.updated_at,
                    "is_active": user.is_active,
                }
            return None

    def create_user(self, email: str) -> dict:
        """Create a new user"""
        with get_session() as session:
            user = User(email=email)
            session.add(user)
            session.flush()  # Get ID, commit handled by context
            logger.info(f"Created new user: {email}")
            return {
                "id": str(user.id) if user.id else None,
                "email": user.email,
                "created_at": user.created_at,
                "updated_at": user.updated_at,
                "is_active": user.is_active,
            }

    def get_or_create_user(self, email: str) -> dict:
        """Get existing user or create new one"""
        user = self.get_user_by_email(email)
        if not user:
            user = self.create_user(email)
        return user

    def update_user(self, user_id: UUID, **kwargs) -> Optional[dict]:
        """Update user attributes"""
        user_id_for_db = handle_uuid_for_db(user_id)
        with get_session() as session:
            user = session.query(User).filter(User.id == user_id_for_db).first()
            if not user:
                return None

            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)

            return {
                "id": str(user.id) if user.id else None,
                "email": user.email,
                "created_at": user.created_at,
                "updated_at": user.updated_at,
                "is_active": user.is_active,
            }

    def delete_user(self, user_id: UUID) -> bool:
        """Delete a user and all their receipts"""
        user_id_for_db = handle_uuid_for_db(user_id)
        with get_session() as session:
            user = session.query(User).filter(User.id == user_id_for_db).first()
            if not user:
                return False

            session.delete(user)  # Will cascade delete receipts
            logger.info(f"Deleted user: {user_id}")
            return True
