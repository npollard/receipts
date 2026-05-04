from decimal import Decimal
from typing import Any


class ReceiptDataMapper:
    """Convert parser output into repository-friendly plain data."""

    def extract(self, parsed: Any) -> Any:
        # Normalize parsed shapes (dict or object with attributes)
        if isinstance(parsed, dict) and 'parsed' in parsed:
            receipt_data = parsed['parsed']
        else:
            receipt_data = getattr(parsed, 'parsed', None) or getattr(parsed, 'receipt_data', None) or parsed

        model_dump = getattr(receipt_data, 'model_dump', None)
        if callable(model_dump):
            receipt_data = model_dump()
        return self.to_plain_data(receipt_data)

    def to_plain_data(self, value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, dict):
            return {key: self.to_plain_data(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.to_plain_data(item) for item in value]
        return value
