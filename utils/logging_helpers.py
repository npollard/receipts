"""Logging utilities and helpers"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from enum import Enum


class LogLevel(Enum):
    """Log levels for easy configuration"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def setup_logging(level: LogLevel = LogLevel.INFO, 
                 format_string: Optional[str] = None,
                 log_file: Optional[Union[str, Path]] = None,
                 console_output: bool = True) -> None:
    """Setup logging configuration
    
    Args:
        level: Logging level
        format_string: Custom format string
        log_file: Optional file to log to
        console_output: Whether to output to console
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level.value)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level.value)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level.value)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str, level: Optional[LogLevel] = None) -> logging.Logger:
    """Get a logger instance with optional level setting
    
    Args:
        name: Logger name
        level: Optional level to set
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(level.value)
    return logger


def log_function_call(func):
    """Decorator to log function calls
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    logger = get_logger(func.__module__)
    
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_method_call(method):
    """Decorator to log method calls
    
    Args:
        method: Method to decorate
        
    Returns:
        Decorated method
    """
    logger = get_logger(method.__module__)
    
    def wrapper(self, *args, **kwargs):
        class_name = self.__class__.__name__
        logger.debug(f"Calling {class_name}.{method.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = method(self, *args, **kwargs)
            logger.debug(f"{class_name}.{method.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{class_name}.{method.__name__} failed with error: {e}")
            raise
    
    return wrapper


def configure_verbose_logging(enable_debug: bool = False) -> None:
    """Configure verbose logging for debugging
    
    Args:
        enable_debug: Whether to enable debug level logging
    """
    level = LogLevel.DEBUG if enable_debug else LogLevel.INFO
    setup_logging(level=level)


def log_performance(func):
    """Decorator to log function performance
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import time
    logger = get_logger(func.__module__)
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper


def create_file_logger(name: str, log_file: Union[str, Path], 
                      level: LogLevel = LogLevel.INFO) -> logging.Logger:
    """Create a logger that only writes to a file
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.value)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create file handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level.value)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger


def suppress_library_logging(library_names: list[str]) -> None:
    """Suppress logging for specified libraries
    
    Args:
        library_names: List of library names to suppress
    """
    for library_name in library_names:
        logging.getLogger(library_name).setLevel(logging.WARNING)


def setup_library_logging() -> None:
    """Setup logging configuration for common libraries"""
    # Suppress verbose logging from common libraries
    suppress_library_logging([
        'urllib3',
        'requests',
        'langchain',
        'openai',
        'httpx',
        'sqlalchemy'
    ])


# Common logging presets
def setup_production_logging(log_file: Union[str, Path] = "app.log") -> None:
    """Setup production logging configuration"""
    setup_logging(
        level=LogLevel.INFO,
        log_file=log_file,
        console_output=False
    )
    setup_library_logging()


def setup_development_logging() -> None:
    """Setup development logging configuration"""
    setup_logging(
        level=LogLevel.DEBUG,
        console_output=True
    )
