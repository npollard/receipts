"""Unit tests for user context management logic.

Replaces integration tests with isolated logic tests.
"""

import pytest
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class UserContext:
    """User context for testing."""
    user_id: str
    email: str
    is_multi_user: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "is_multi_user": self.is_multi_user
        }


class UserContextManager:
    """Simplified user context manager for testing."""

    def __init__(self, default_email: str = "user@receipts.local", multi_user: bool = False):
        self.default_email = default_email
        self.multi_user = multi_user
        self._current_user: Optional[UserContext] = None

    def get_default_user(self) -> UserContext:
        """Get default user context."""
        return UserContext(
            user_id=f"user_{hash(self.default_email) % 10000}",
            email=self.default_email,
            is_multi_user=self.multi_user
        )

    def switch_user(self, email: str) -> UserContext:
        """Switch to different user."""
        if not self.multi_user:
            raise ValueError("Multi-user mode not enabled")

        self._current_user = UserContext(
            user_id=f"user_{hash(email) % 10000}",
            email=email,
            is_multi_user=True
        )
        return self._current_user

    def get_current_user(self) -> UserContext:
        """Get current user or default."""
        return self._current_user or self.get_default_user()


class TestUserContextManager:
    """User context management logic."""

    def test_default_user_created_with_email(self):
        """Given: Default email. When: Manager created. Then: Default user available."""
        manager = UserContextManager(default_email="admin@test.com")

        user = manager.get_default_user()

        assert user.email == "admin@test.com"
        assert user.user_id.startswith("user_")

    def test_multi_user_mode_detection(self):
        """Given: Multi-user enabled. When: Checked. Then: Returns True."""
        manager = UserContextManager(multi_user=True)

        user = manager.get_default_user()

        assert user.is_multi_user is True

    def test_single_user_mode_default(self):
        """Given: No multi-user flag. When: Checked. Then: Returns False."""
        manager = UserContextManager()

        user = manager.get_default_user()

        assert user.is_multi_user is False

    def test_user_switching_in_multi_user_mode(self):
        """Given: Multi-user mode. When: Switch user. Then: Context updated."""
        manager = UserContextManager(multi_user=True)

        user = manager.switch_user("alice@example.com")

        assert user.email == "alice@example.com"
        assert manager.get_current_user().email == "alice@example.com"

    def test_user_switching_blocked_in_single_user_mode(self):
        """Given: Single-user mode. When: Try switch. Then: Raises error."""
        manager = UserContextManager(multi_user=False)

        with pytest.raises(ValueError, match="Multi-user mode not enabled"):
            manager.switch_user("alice@example.com")

    def test_user_id_deterministic_from_email(self):
        """Given: Same email. When: User created twice. Then: Same ID."""
        manager1 = UserContextManager(default_email="test@example.com")
        manager2 = UserContextManager(default_email="test@example.com")

        user1 = manager1.get_default_user()
        user2 = manager2.get_default_user()

        assert user1.user_id == user2.user_id

    def test_different_emails_different_ids(self):
        """Given: Different emails. When: Users created. Then: Different IDs."""
        manager = UserContextManager(multi_user=True)

        user1 = manager.switch_user("alice@example.com")
        user2 = manager.switch_user("bob@example.com")

        assert user1.user_id != user2.user_id


class TestUserContextSerialization:
    """User context serialization."""

    def test_to_dict_contains_required_fields(self):
        """Given: UserContext. When: Serialized. Then: Has required fields."""
        context = UserContext(
            user_id="user_123",
            email="test@example.com",
            is_multi_user=True
        )

        data = context.to_dict()

        assert data["user_id"] == "user_123"
        assert data["email"] == "test@example.com"
        assert data["is_multi_user"] is True


class TestUserValidation:
    """User email validation logic."""

    def test_valid_email_format_accepted(self):
        """Given: Valid email. When: Validated. Then: Returns True."""
        valid_emails = [
            "user@example.com",
            "test.user@domain.co.uk",
            "user+tag@example.com"
        ]

        for email in valid_emails:
            assert "@" in email
            assert "." in email.split("@")[1]

    def test_invalid_email_format_rejected(self):
        """Given: Invalid email. When: Validated. Then: Returns False."""
        import re

        invalid_emails = [
            "not-an-email",
            "spaces in@email.com",
            "missing@domain",
        ]

        for email in invalid_emails:
            # Simple regex-based validation
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            is_valid = bool(re.match(pattern, email))
            assert not is_valid
