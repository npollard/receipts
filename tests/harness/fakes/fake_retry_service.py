"""Fake retry service for deterministic retry behavior testing."""

from dataclasses import dataclass
from typing import Callable, Any, List, Optional, Type, Dict
from enum import Enum, auto
import time

from .fake_component import FakeComponent, ConfigurationError


class RetryStrategy(Enum):
    """Retry strategies supported by the fake."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    LLM_SELF_CORRECTION = "llm_self_correction"
    RAG_FALLBACK = "rag_fallback"
    VISION_REPARSE = "vision_reparse"


@dataclass
class RetryAttempt:
    """Record of a retry attempt."""
    attempt_number: int
    strategy: str
    exception_type: Optional[str]
    duration_ms: float
    succeeded: bool


class FakeRetryService(FakeComponent):
    """Fake retry service with deterministic retry behavior.
    
    Simulates:
    - Configurable success on Nth attempt
    - Strategy selection based on error type
    - Immediate retries (no actual delays in tests)
    - Retry attempt recording
    
    Example:
        >>> retry = FakeRetryService()
        >>> retry.set_succeed_on_attempt(2)  # Fail once, then succeed
        >>> retry.set_strategy_for_error(ValidationError, "LLM_SELF_CORRECTION")
        >>> result = retry.execute_with_retry(func, args)
        >>> retry.get_attempt_count()  # 2
    """
    
    def __init__(self):
        super().__init__()
        self._max_retries: int = 3
        self._succeed_on_attempt: int = 1
        self._current_attempt: int = 0
        self._strategies: Dict[Type[Exception], str] = {}
        self._default_strategy: str = RetryStrategy.EXPONENTIAL_BACKOFF.value
        self._attempt_history: List[RetryAttempt] = []
        self._should_delay: bool = False  # Never delay in tests
    
    def set_max_retries(self, n: int) -> "FakeRetryService":
        """Set maximum number of retry attempts."""
        self._max_retries = n
        return self
    
    def set_succeed_on_attempt(self, n: int) -> "FakeRetryService":
        """Configure to succeed on the Nth attempt (1-indexed).
        
        Args:
            n: Which attempt should succeed (1 = first attempt succeeds)
        """
        self._succeed_on_attempt = n
        return self
    
    def set_strategy_for_error(
        self,
        error_type: Type[Exception],
        strategy: str
    ) -> "FakeRetryService":
        """Configure retry strategy for specific error type."""
        self._strategies[error_type] = strategy
        return self
    
    def set_default_strategy(self, strategy: str) -> "FakeRetryService":
        """Set default strategy when no specific match."""
        self._default_strategy = strategy
        return self
    
    def reset_attempts(self) -> None:
        """Reset attempt counter for fresh test."""
        self._current_attempt = 0
        self._attempt_history.clear()
    
    def _select_strategy(self, exception: Optional[Exception]) -> str:
        """Select retry strategy based on exception type."""
        if exception is None:
            return self._default_strategy
        
        for error_type, strategy in self._strategies.items():
            if isinstance(exception, error_type):
                return strategy
        
        return self._default_strategy
    
    def execute_with_retry(
        self,
        func: Callable,
        *args,
        error_types: Optional[tuple] = None,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        **kwargs
    ) -> Any:
        """Execute a function with configured retry behavior.
        
        Args:
            func: Function to execute
            *args: Function positional arguments
            error_types: Tuple of exception types to catch
            on_retry: Callback on each retry
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries exhausted
        """
        start = time.time()
        self._current_attempt = 0
        last_exception: Optional[Exception] = None
        
        while self._current_attempt < self._max_retries:
            self._current_attempt += 1
            attempt_start = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful attempt
                attempt = RetryAttempt(
                    attempt_number=self._current_attempt,
                    strategy=self._select_strategy(last_exception),
                    exception_type=None,
                    duration_ms=(time.time() - attempt_start) * 1000,
                    succeeded=True
                )
                self._attempt_history.append(attempt)
                
                total_duration_ms = (time.time() - start) * 1000
                self._record_call(
                    "execute_with_retry",
                    (func,) + args,
                    kwargs,
                    result=result,
                    duration_ms=total_duration_ms
                )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should succeed on this attempt anyway
                if self._current_attempt >= self._succeed_on_attempt:
                    # Force success by not re-raising
                    attempt = RetryAttempt(
                        attempt_number=self._current_attempt,
                        strategy=self._select_strategy(e),
                        exception_type=type(e).__name__,
                        duration_ms=(time.time() - attempt_start) * 1000,
                        succeeded=True  # Forced success
                    )
                    self._attempt_history.append(attempt)
                    
                    # Return a mock success result
                    total_duration_ms = (time.time() - start) * 1000
                    mock_result = {"forced_success": True, "original_error": str(e)}
                    
                    self._record_call(
                        "execute_with_retry",
                        (func,) + args,
                        kwargs,
                        result=mock_result,
                        duration_ms=total_duration_ms
                    )
                    
                    return mock_result
                
                # Record failed attempt
                attempt = RetryAttempt(
                    attempt_number=self._current_attempt,
                    strategy=self._select_strategy(e),
                    exception_type=type(e).__name__,
                    duration_ms=(time.time() - attempt_start) * 1000,
                    succeeded=False
                )
                self._attempt_history.append(attempt)
                
                # Call retry callback if provided
                if on_retry:
                    on_retry(self._current_attempt, e)
                
                # Check if we should continue retrying
                if self._current_attempt >= self._max_retries:
                    break
        
        # All retries exhausted
        total_duration_ms = (time.time() - start) * 1000
        self._record_call(
            "execute_with_retry",
            (func,) + args,
            kwargs,
            exception=last_exception,
            duration_ms=total_duration_ms
        )
        raise last_exception
    
    def get_attempt_count(self) -> int:
        """Get number of attempts made in last execution."""
        return len(self._attempt_history)
    
    def get_strategies_used(self) -> List[str]:
        """Get list of strategies used in last execution."""
        return [a.strategy for a in self._attempt_history]
    
    def was_retried(self) -> bool:
        """Check if any retries occurred (more than 1 attempt)."""
        return len(self._attempt_history) > 1
    
    def get_attempt_history(self) -> List[RetryAttempt]:
        """Get full history of retry attempts."""
        return list(self._attempt_history)
    
    def succeeded_on_attempt(self, n: int) -> bool:
        """Check if execution succeeded specifically on attempt N."""
        for attempt in self._attempt_history:
            if attempt.attempt_number == n and attempt.succeeded:
                return True
        return False
    
    def used_strategy(self, strategy: str) -> bool:
        """Check if specific strategy was used."""
        return any(a.strategy == strategy for a in self._attempt_history)
