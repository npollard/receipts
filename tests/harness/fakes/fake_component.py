"""Base class for all fake components with call tracking."""

from dataclasses import dataclass, field
from typing import Any, List, Dict, Callable, Optional, Type
from datetime import datetime
import time


@dataclass
class CallRecord:
    """Record of a method call on a fake component."""
    method: str
    args: tuple
    kwargs: Dict[str, Any]
    result: Any = None
    exception: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0


class FakeComponent:
    """Base class for fake components with call recording and configuration.
    
    All fakes inherit from this to get:
    - Call history tracking
    - Configurable behaviors per method
    - Timing information
    - Reset capability
    """
    
    def __init__(self):
        self._calls: List[CallRecord] = []
        self._behaviors: Dict[str, Callable] = {}
        self._default_behaviors: Dict[str, Callable] = {}
    
    def _record_call(
        self,
        method: str,
        args: tuple,
        kwargs: Dict[str, Any],
        result: Any = None,
        exception: Optional[Exception] = None,
        duration_ms: float = 0.0
    ) -> None:
        """Record a method call."""
        record = CallRecord(
            method=method,
            args=args,
            kwargs=kwargs,
            result=result,
            exception=exception,
            duration_ms=duration_ms
        )
        self._calls.append(record)
    
    def configure(self, method: str, behavior: Callable) -> "FakeComponent":
        """Configure a custom behavior for a method.
        
        Args:
            method: Method name to configure
            behavior: Callable that will be invoked instead of default
        
        Returns:
            Self for chaining
        """
        self._behaviors[method] = behavior
        return self
    
    def _get_behavior(self, method: str) -> Optional[Callable]:
        """Get the configured behavior for a method, or None."""
        return self._behaviors.get(method, self._default_behaviors.get(method))
    
    def get_calls(self, method: Optional[str] = None) -> List[CallRecord]:
        """Get recorded calls, optionally filtered by method name.
        
        Args:
            method: Optional method name to filter by
            
        Returns:
            List of call records
        """
        if method is None:
            return list(self._calls)
        return [c for c in self._calls if c.method == method]
    
    def get_call_count(self, method: Optional[str] = None) -> int:
        """Get number of calls, optionally filtered by method name."""
        return len(self.get_calls(method))
    
    def was_called(self, method: Optional[str] = None) -> bool:
        """Check if any calls were made."""
        return self.get_call_count(method) > 0
    
    def reset(self) -> None:
        """Reset all call history and configured behaviors."""
        self._calls.clear()
        self._behaviors.clear()
    
    def set_default_behavior(self, method: str, behavior: Callable) -> None:
        """Set a default behavior for a method (used when no explicit config)."""
        self._default_behaviors[method] = behavior


class FakeError(Exception):
    """Base exception for fake-related errors."""
    pass


class ConfigurationError(FakeError):
    """Raised when fake is not properly configured."""
    pass
