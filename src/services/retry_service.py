"""
Retry Service (Rewritten)

Key properties:
- Exception-driven retry (NOT APIResponse.success-based)
- Single responsibility: execute + track attempts
- No nested retry loops
- Explicit attempt tracking for test harness
- Strategy tracking based on exception types
"""

import logging
import time
from typing import Callable, Optional, Tuple, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    EXPONENTIAL_BACKOFF = "EXPONENTIAL_BACKOFF"
    LINEAR_BACKOFF = "LINEAR_BACKOFF"
    FIXED_DELAY = "FIXED_DELAY"
    IMMEDIATE = "IMMEDIATE"
    LLM_SELF_CORRECTION = "LLM_SELF_CORRECTION"


class RetryService:
    """
    Exception-driven retry executor.

    Responsibilities:
    - Execute function with retries
    - Track attempts + strategies

    NOT responsible for:
    - Deciding *whether* to retry (caller responsibility)
    """

    def __init__(
        self,
        max_retries: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay: float = 0.0,
        max_delay: float = 5.0,
        backoff_multiplier: float = 2.0,
    ):
        self.max_retries = max_retries
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier

        # Tracking
        self._attempt_history: List[dict] = []
        self._strategies_used: List[str] = []

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def execute_with_retry(
        self,
        func: Callable,
        *args,
        error_types: Optional[Tuple[type, ...]] = None,
        **kwargs,
    ) -> Any:
        """
        Execute function with retries.

        Exception-driven:
        - success = no exception
        - failure = exception
        """

        if error_types is None:
            error_types = (Exception,)

        self._reset_tracking()

        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = func(*args, **kwargs)

                self._record_attempt(attempt, success=True)
                return result

            except error_types as e:
                last_exception = e

                strategy = self._select_strategy(e)
                self._record_attempt(attempt, success=False, strategy=strategy, error=e)

                if attempt == self.max_retries:
                    break

                self._wait(attempt)

        # Final failure
        raise last_exception

    # --------------------------------------------------
    # Tracking
    # --------------------------------------------------

    def _reset_tracking(self):
        self._attempt_history = []
        self._strategies_used = []

    def _record_attempt(self, attempt: int, success: bool, strategy: Optional[str] = None, error: Exception = None):
        entry = {
            "attempt": attempt,
            "success": success,
        }

        if strategy:
            entry["strategy"] = strategy
            self._strategies_used.append(strategy)

        if error:
            entry["error"] = str(error)

        self._attempt_history.append(entry)

    def get_attempt_count(self) -> int:
        return len(self._attempt_history)

    def get_attempt_history(self) -> List[dict]:
        return self._attempt_history

    def get_strategies_used(self) -> List[str]:
        return self._strategies_used

    # --------------------------------------------------
    # Strategy
    # --------------------------------------------------

    def _select_strategy(self, exception: Exception) -> str:
        """
        Map exception → retry strategy
        """

        if isinstance(exception, ValueError):
            return RetryStrategy.LLM_SELF_CORRECTION.value

        return self.strategy.value

    # --------------------------------------------------
    # Backoff
    # --------------------------------------------------

    def _wait(self, attempt: int):
        if self.strategy == RetryStrategy.IMMEDIATE:
            return

        if self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = min(self.base_delay * attempt, self.max_delay)
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(self.base_delay * (self.backoff_multiplier ** (attempt - 1)), self.max_delay)
        else:
            delay = self.base_delay

        if delay > 0:
            time.sleep(delay)


# Default instance

default_retry_service = RetryService()
