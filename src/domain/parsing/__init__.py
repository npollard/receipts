"""Domain parsing module for receipt processing"""

from .receipt_parser import ReceiptParser
from .ai_parsing import ReceiptParser as AIParser

__all__ = [
    "ReceiptParser",
    "AIParser",
]
