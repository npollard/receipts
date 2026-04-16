"""Retry service tests - pure unit tests for isolated retry logic."""

import pytest
from unittest.mock import Mock, patch
from services.retry_service import RetryService, RetryStrategy
from api_response import APIResponse


class TestRetryServiceConfiguration:
    """Retry service configuration scenarios."""

    def test_default_configuration(self):
        """Given: No config. When: Created. Then: Uses defaults."""
        service = RetryService()

        assert service.max_retries == 3
        assert service.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert service.base_delay == 0.0

    def test_custom_max_retries(self):
        """Given: Custom max_retries. When: Created. Then: Uses custom value."""
        service = RetryService(max_retries=5)

        assert service.max_retries == 5

    def test_custom_strategy(self):
        """Given: Custom strategy. When: Created. Then: Uses custom strategy."""
        service = RetryService(strategy=RetryStrategy.FIXED_DELAY)

        assert service.strategy == RetryStrategy.FIXED_DELAY

    def test_custom_delays(self):
        """Given: Custom delays. When: Created. Then: Uses custom values."""
        service = RetryService(base_delay=2.0, max_delay=30.0, backoff_multiplier=3.0)

        assert service.base_delay == 2.0
        assert service.max_delay == 30.0
        assert service.backoff_multiplier == 3.0


class TestRetryServiceExecution:
    """Retry execution behavior scenarios."""

    def test_success_no_retry_needed(self):
        """Given: Function succeeds. When: Executed. Then: Returns result immediately."""
        service = RetryService()

        def success_func():
            return {"data": "value"}

        result = service.execute_with_retry(success_func)

        assert result == {"data": "value"}

    def test_retry_then_success(self):
        """Given: Function fails once then succeeds. When: Executed. Then: Retries and succeeds."""
        service = RetryService(max_retries=3, strategy=RetryStrategy.IMMEDIATE)

        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return {"attempt": call_count}

        result = service.execute_with_retry(flaky_func, error_types=(ValueError,))

        assert result == {"attempt": 2}
        assert call_count == 2

    def test_retry_exhausted_failure(self):
        """Given: Function always fails. When: Executed. Then: Raises exception after max retries."""
        service = RetryService(max_retries=2, strategy=RetryStrategy.IMMEDIATE)

        def always_fails():
            raise ValueError("Persistent error")

        with pytest.raises(ValueError, match="Persistent error"):
            service.execute_with_retry(always_fails, error_types=(ValueError,))

    def test_no_retry_for_unspecified_error_types(self):
        """Given: Different error type. When: Executed. Then: Exception propagates, no retry."""
        service = RetryService(max_retries=3, strategy=RetryStrategy.IMMEDIATE)

        call_count = 0

        def raises_wrong_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Wrong error type")

        # When error type doesn't match, exception propagates immediately
        with pytest.raises(TypeError):
            service.execute_with_retry(
                raises_wrong_error,
                error_types=(ValueError,)  # Not TypeError
            )

        assert call_count == 1  # Called once, no retries

    def test_retry_attempts_tracked(self):
        """Given: Retries occur. When: Executed. Then: Attempts are tracked."""
        service = RetryService(max_retries=3, strategy=RetryStrategy.IMMEDIATE)

        def always_fails():
            raise ValueError("Error")

        with pytest.raises(ValueError):
            service.execute_with_retry(always_fails, error_types=(ValueError,))

        # Check that all attempts were tracked
        assert service.get_attempt_count() == 3


class TestRetryStrategies:
    """Retry strategy behavior scenarios (testing via _wait)."""

    def test_exponential_backoff_waits_correctly(self):
        """Given: Exponential strategy. When: Retry wait. Then: Exponential delay."""
        service = RetryService(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=10.0
        )

        with patch('time.sleep') as mock_sleep:
            service._wait(1)  # After 1st attempt (attempt=1)
            assert mock_sleep.call_args[0][0] == 1.0  # base_delay * (2^0)

            service._wait(2)  # After 2nd attempt
            assert mock_sleep.call_args[0][0] == 2.0  # base_delay * (2^1)

            service._wait(5)  # After 5th attempt - capped
            assert mock_sleep.call_args[0][0] == 10.0  # max_delay

    def test_linear_backoff_waits_correctly(self):
        """Given: Linear strategy. When: Retry wait. Then: Linear delay."""
        service = RetryService(
            strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay=1.0
        )

        with patch('time.sleep') as mock_sleep:
            service._wait(1)
            assert mock_sleep.call_args[0][0] == 1.0

            service._wait(2)
            assert mock_sleep.call_args[0][0] == 2.0

    def test_fixed_delay_constant_wait(self):
        """Given: Fixed strategy. When: Retry wait. Then: Constant delay."""
        service = RetryService(
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay=2.0
        )

        with patch('time.sleep') as mock_sleep:
            service._wait(1)
            assert mock_sleep.call_args[0][0] == 2.0

            service._wait(5)
            assert mock_sleep.call_args[0][0] == 2.0  # Always 2.0

    def test_immediate_strategy_no_wait(self):
        """Given: Immediate strategy. When: Retry wait. Then: No sleep."""
        service = RetryService(strategy=RetryStrategy.IMMEDIATE)

        with patch('time.sleep') as mock_sleep:
            service._wait(5)
            mock_sleep.assert_not_called()


class TestRetryServiceEdgeCases:
    """Edge case scenarios."""

    def test_zero_max_retries(self):
        """Given: Zero retries. When: Function called. Then: Function not executed, raises error."""
        service = RetryService(max_retries=0)

        call_count = 0

        def never_called():
            nonlocal call_count
            call_count += 1
            return {"data": "value"}

        # With 0 retries, loop doesn't execute, raises None (edge case behavior)
        with pytest.raises(Exception):
            service.execute_with_retry(never_called)

        # With 0 retries, function is never called
        assert call_count == 0

    def test_function_arguments_passed(self):
        """Given: Function with args. When: Executed. Then: Args passed correctly."""
        service = RetryService()

        def func_with_args(a, b, c=None):
            return {"a": a, "b": b, "c": c}

        result = service.execute_with_retry(func_with_args, 1, 2, c=3)

        assert result == {"a": 1, "b": 2, "c": 3}

    def test_exception_message_in_error(self):
        """Given: Function raises. When: Retries exhausted. Then: Exception is raised with message."""
        service = RetryService(max_retries=1, strategy=RetryStrategy.IMMEDIATE)

        def raises_specific():
            raise ValueError("Specific error message")

        with pytest.raises(ValueError, match="Specific error message"):
            service.execute_with_retry(raises_specific, error_types=(ValueError,))
