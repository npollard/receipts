"""Retry service for handling retry logic with error handling"""

import logging
from typing import Callable, Any, Optional, Type, Union
from functools import wraps
import time
from enum import Enum

from api_response import APIResponse

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies for different scenarios"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


class RetryService:
    """Service for handling retry logic with configurable strategies"""

    def __init__(self, 
                 max_retries: int = 3,
                 strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0):
        """
        Initialize retry service
        
        Args:
            max_retries: Maximum number of retry attempts
            strategy: Retry strategy to use
            base_delay: Base delay between retries (in seconds)
            max_delay: Maximum delay between retries
            backoff_multiplier: Multiplier for exponential backoff
        """
        self.max_retries = max_retries
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        
        logger.info(f"Initialized RetryService: max_retries={max_retries}, strategy={strategy.value}")

    def execute_with_retry(self, 
                          func: Callable,
                          *args,
                          error_types: Optional[tuple] = None,
                          on_retry: Optional[Callable[[int, Exception], None]] = None,
                          **kwargs) -> APIResponse:
        """
        Execute a function with retry logic
        
        Args:
            func: Function to execute
            *args: Function arguments
            error_types: Tuple of exception types to retry on
            on_retry: Callback function called on each retry (attempt, exception)
            **kwargs: Function keyword arguments
            
        Returns:
            APIResponse from the last function call
        """
        if error_types is None:
            error_types = (Exception,)
            
        last_exception = None
        last_result = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{self.max_retries}")
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Check if result indicates success
                if hasattr(result, 'success') and result.success:
                    logger.info(f"Success on attempt {attempt}")
                    return result
                else:
                    # Treat non-success results as failures for retry
                    error_msg = getattr(result, 'error', 'Function returned non-success result')
                    last_exception = Exception(error_msg)
                    last_result = result
                    
                    if attempt < self.max_retries:
                        logger.warning(f"Attempt {attempt} failed: {error_msg}")
                        if on_retry:
                            on_retry(attempt, last_exception)
                        self._wait_before_retry(attempt)
                    else:
                        logger.error(f"All {self.max_retries} attempts failed")
                        return result
                        
            except error_types as e:
                last_exception = e
                logger.warning(f"Attempt {attempt} failed with {type(e).__name__}: {str(e)}")
                
                if attempt < self.max_retries:
                    if on_retry:
                        on_retry(attempt, e)
                    self._wait_before_retry(attempt)
                else:
                    logger.error(f"All {self.max_retries} attempts failed with {type(e).__name__}")
                    break
        
        # Return failure result if we get here
        if last_result and hasattr(last_result, 'success'):
            return last_result
        else:
            error_msg = f"All {self.max_retries} attempts failed"
            if last_exception:
                error_msg += f": {str(last_exception)}"
            return APIResponse.failure(error_msg)

    def execute_with_retry_and_fix(self,
                                  func: Callable,
                                  fix_func: Callable[[str, Exception, int], str],
                                  *args,
                                  error_types: Optional[tuple] = None,
                                  **kwargs) -> APIResponse:
        """
        Execute function with retry logic and error fixing
        
        Args:
            func: Function to execute
            fix_func: Function to generate fix prompts (input_text, error, attempt)
            *args: Function arguments
            error_types: Tuple of exception types to retry on
            **kwargs: Function keyword arguments
            
        Returns:
            APIResponse from the last function call
        """
        if error_types is None:
            error_types = (Exception,)
            
        last_exception = None
        last_result = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{self.max_retries}")
                
                # For attempts > 1, try to fix the input using the fix function
                if attempt > 1 and last_exception and fix_func:
                    try:
                        # Generate fixed input
                        fixed_input = fix_func(*args, last_exception, attempt, **kwargs)
                        # Replace the first argument with the fixed input
                        if args:
                            args = (fixed_input,) + args[1:]
                        else:
                            # If no args, try to fix kwargs
                            if 'text' in kwargs:
                                kwargs['text'] = fixed_input
                    except Exception as fix_error:
                        logger.warning(f"Fix attempt {attempt} failed: {str(fix_error)}")
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Check if result indicates success
                if hasattr(result, 'success') and result.success:
                    logger.info(f"Success on attempt {attempt}")
                    return result
                else:
                    # Treat non-success results as failures for retry
                    error_msg = getattr(result, 'error', 'Function returned non-success result')
                    last_exception = Exception(error_msg)
                    last_result = result
                    
                    if attempt < self.max_retries:
                        logger.warning(f"Attempt {attempt} failed: {error_msg}")
                        self._wait_before_retry(attempt)
                    else:
                        logger.error(f"All {self.max_retries} attempts failed")
                        return result
                        
            except error_types as e:
                last_exception = e
                logger.warning(f"Attempt {attempt} failed with {type(e).__name__}: {str(e)}")
                
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                else:
                    logger.error(f"All {self.max_retries} attempts failed with {type(e).__name__}")
                    break
        
        # Return failure result if we get here
        if last_result and hasattr(last_result, 'success'):
            return last_result
        else:
            error_msg = f"All {self.max_retries} attempts failed"
            if last_exception:
                error_msg += f": {str(last_exception)}"
            return APIResponse.failure(error_msg)

    def _wait_before_retry(self, attempt: int):
        """Wait before retry based on strategy"""
        if self.strategy == RetryStrategy.IMMEDIATE:
            return
        elif self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = min(self.base_delay * attempt, self.max_delay)
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(self.base_delay * (self.backoff_multiplier ** (attempt - 1)), self.max_delay)
        else:
            delay = self.base_delay
            
        logger.debug(f"Waiting {delay:.2f}s before retry attempt {attempt + 1}")
        time.sleep(delay)

    def retry_decorator(self, 
                      error_types: Optional[tuple] = None,
                      on_retry: Optional[Callable[[int, Exception], None]] = None):
        """
        Decorator for adding retry logic to functions
        
        Args:
            error_types: Tuple of exception types to retry on
            on_retry: Callback function called on each retry
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_retry(
                    func, *args, 
                    error_types=error_types, 
                    on_retry=on_retry,
                    **kwargs
                )
            return wrapper
        return decorator


# Default retry service instance for common use
default_retry_service = RetryService()


def retry_with_default(max_retries: int = 3, 
                     strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
                     error_types: Optional[tuple] = None):
    """
    Decorator using default retry service
    
    Args:
        max_retries: Maximum number of retry attempts
        strategy: Retry strategy to use
        error_types: Tuple of exception types to retry on
        
    Returns:
        Decorated function
    """
    service = RetryService(max_retries=max_retries, strategy=strategy)
    return service.retry_decorator(error_types=error_types)
