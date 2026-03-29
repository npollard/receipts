"""API response utilities"""

from typing import Any


class APIResponse:
    """Standardized API response structure"""
    
    def __init__(self, status: str, data: Any = None, error: str = ""):
        self.status = status  # "success" or "failed"
        self.data = data
        self.error = error

    @classmethod
    def success(cls, data: Any) -> 'APIResponse':
        """Create a successful response"""
        return cls(status="success", data=data)

    @classmethod
    def failure(cls, error: str) -> 'APIResponse':
        """Create a failed response"""
        return cls(status="failed", error=error, data=None)
