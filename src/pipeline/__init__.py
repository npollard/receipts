"""Pipeline module for receipt processing"""

from .processor import (
    Processor,
    ReceiptProcessor,  # Alias for backwards compatibility
)

__all__ = [
    'Processor',
    'ReceiptProcessor',
]
