"""Centralized logging configuration with structured prefixes"""

import logging
import sys
from typing import Optional
from enum import Enum


class LogPrefix(Enum):
    """Log prefixes for different components"""
    OCR = "[OCR]"
    PARSER = "[PARSER]"
    VALIDATION = "[VALIDATION]"
    SCORING = "[SCORING]"
    STORAGE = "[STORAGE]"
    BATCH = "[BATCH]"
    TOKEN = "[TOKEN]"
    FILE = "[FILE]"
    PIPELINE = "[PIPELINE]"
    RETRY = "[RETRY]"
    CORE = "[CORE]"
    INFRA = "[INFRA]"
    MIGRATION = "[MIGRATION]"


class StructuredLogger:
    """Structured logger with consistent prefixes and formatting"""
    
    def __init__(self, name: str, prefix: LogPrefix):
        self.logger = logging.getLogger(name)
        self.prefix = prefix.value
        
    def _format_message(self, message: str) -> str:
        """Format message with prefix"""
        return f"{self.prefix} {message}"
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(self._format_message(message), **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(self._format_message(message), **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(self._format_message(message), **kwargs)


def get_logger(name: str, prefix: LogPrefix) -> StructuredLogger:
    """Get a structured logger with the specified prefix"""
    return StructuredLogger(name, prefix)


def setup_logging(level: str = "INFO", format_string: Optional[str] = None):
    """Setup centralized logging configuration"""
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress excessive logging from third-party libraries
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Set specific loggers to appropriate levels
    logging.getLogger("receipts").setLevel(logging.INFO)
    logging.getLogger("services").setLevel(logging.INFO)
    logging.getLogger("domain").setLevel(logging.INFO)
    logging.getLogger("pipeline").setLevel(logging.INFO)
    logging.getLogger("storage").setLevel(logging.INFO)


# Convenience functions for common prefixes
def get_ocr_logger(name: str) -> StructuredLogger:
    """Get OCR logger"""
    return get_logger(name, LogPrefix.OCR)


def get_parser_logger(name: str) -> StructuredLogger:
    """Get parser logger"""
    return get_logger(name, LogPrefix.PARSER)


def get_validation_logger(name: str) -> StructuredLogger:
    """Get validation logger"""
    return get_logger(name, LogPrefix.VALIDATION)


def get_scoring_logger(name: str) -> StructuredLogger:
    """Get scoring logger"""
    return get_logger(name, LogPrefix.SCORING)


def get_storage_logger(name: str) -> StructuredLogger:
    """Get storage logger"""
    return get_logger(name, LogPrefix.STORAGE)


def get_batch_logger(name: str) -> StructuredLogger:
    """Get batch logger"""
    return get_logger(name, LogPrefix.BATCH)


def get_token_logger(name: str) -> StructuredLogger:
    """Get token logger"""
    return get_logger(name, LogPrefix.TOKEN)


def get_file_logger(name: str) -> StructuredLogger:
    """Get file logger"""
    return get_logger(name, LogPrefix.FILE)


def get_pipeline_logger(name: str) -> StructuredLogger:
    """Get pipeline logger"""
    return get_logger(name, LogPrefix.PIPELINE)


def get_retry_logger(name: str) -> StructuredLogger:
    """Get retry logger"""
    return get_logger(name, LogPrefix.RETRY)


def get_core_logger(name: str) -> StructuredLogger:
    """Get core logger"""
    return get_logger(name, LogPrefix.CORE)


def get_infra_logger(name: str) -> StructuredLogger:
    """Get infrastructure logger"""
    return get_logger(name, LogPrefix.INFRA)


def get_migration_logger(name: str) -> StructuredLogger:
    """Get migration logger"""
    return get_logger(name, LogPrefix.MIGRATION)
