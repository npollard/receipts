"""Domain parsing module for receipt processing"""

from .receipt_parser import ReceiptParser
from .parsing_result import ParsingResult

# Alias for backwards compatibility
AIParser = ReceiptParser

__all__ = [
    "ReceiptParser",
    "ParsingResult",
    "AIParser",
]
