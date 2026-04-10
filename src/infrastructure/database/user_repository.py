"""User repository for database operations using unified session management"""

import logging
from typing import Optional
from uuid import UUID

from .session import get_session, get_read_session
from .repository import handle_uuid_for_db
from .models import User
from .mappers import user_to_dto
from shared.models.user_dto import UserDTO

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for user database operations"""

    def __init__(self, database_url: Optional[str] = None):
        # database_url kept for backward compatibility but ignored
        pass

    def get_user_by_email(self, email: str) -> Optional[UserDTO]:
        """Get user by email"""
        with get_read_session() as session:
            user = session.query(User).filter(User.email == email).first()
            if user:
                return user_to_dto(user)
            return None

    def get_user_by_id(self, user_id: UUID) -> Optional[UserDTO]:
        """Get user by ID"""
        user_id_for_db = handle_uuid_for_db(user_id)
        with get_read_session() as session:
            user = session.query(User).filter(User.id == user_id_for_db).first()
            if user:
                return user_to_dto(user)
            return None

    def create_user(self, email: str) -> UserDTO:
        """Create a new user"""
        with get_session() as session:
            user = User(email=email)
            session.add(user)
            session.flush()  # Get ID, commit handled by context
            logger.info(f"Created new user: {email}")
            return user_to_dto(user)

    def get_or_create_user(self, email: str) -> UserDTO:
        """Get existing user or create new one"""
        user = self.get_user_by_email(email)
        if not user:
            user = self.create_user(email)
        return user

    def update_user(self, user_id: UUID, **kwargs) -> Optional[UserDTO]:
        """Update user attributes"""
        user_id_for_db = handle_uuid_for_db(user_id)
        with get_session() as session:
            user = session.query(User).filter(User.id == user_id_for_db).first()
            if not user:
                return None

            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)

            return user_to_dto(user)

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
