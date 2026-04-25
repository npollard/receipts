from decimal import Decimal
from typing import Any


class ReceiptDataMapper:
    """Convert parser output into repository-friendly plain data."""

    def extract(self, parsed: Any) -> Any:
        if hasattr(parsed, "parsed"):
            receipt_data = parsed.parsed
        elif hasattr(parsed, "receipt_data"):
            receipt_data = parsed.receipt_data
        else:
            receipt_data = parsed

        if hasattr(receipt_data, "model_dump"):
            receipt_data = receipt_data.model_dump()
        return self.to_plain_data(receipt_data)

    def to_plain_data(self, value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, dict):
            return {key: self.to_plain_data(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.to_plain_data(item) for item in value]
        return value
