"""Fake receipt parser for deterministic LLM parsing testing."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union
from decimal import Decimal
import json
import time

from .fake_component import FakeComponent, ConfigurationError


@dataclass
class ParserOutput:
    """Structured output from receipt parsing."""
    merchant: str = ""
    date: str = ""
    total: Optional[float] = None
    items: List[Dict[str, Any]] = field(default_factory=list)
    currency: str = "USD"
    tax: Optional[float] = None
    raw_json: Optional[str] = None
    confidence: float = 1.0


class FakeReceiptParser(FakeComponent):
    """Fake receipt parser that returns configurable structured data.

    Simulates:
    - Successful parsing with structured output
    - JSON parsing failures
    - Missing required fields
    - Validation failures
    - Retry strategies used

    Example:
        >>> parser = FakeReceiptParser()
        >>> parser.set_parse_result(
        ...     merchant="Grocery Store",
        ...     total=25.50,
        ...     date="2024-01-15"
        ... )
        >>> result = parser.parse_text("OCR text here")
        >>> result.data["merchant"]  # "Grocery Store"
    """

    def __init__(self):
        super().__init__()
        self._outputs: List[Union[ParserOutput, Exception]] = []
        self._output_index: int = 0
        self._token_usage: Dict[str, int] = {"input": 0, "output": 0}
        self._retry_strategies: List[str] = []
        self._default_output: Optional[ParserOutput] = None
        self._attempt_history: list = []  # Track each attempt with details

    def set_parse_result(
        self,
        merchant: str = "",
        total: Optional[float] = None,
        date: str = "",
        items: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> "FakeReceiptParser":
        """Configure successful parse output.

        Args:
            merchant: Store/merchant name
            total: Receipt total amount
            date: Receipt date string
            items: List of item dicts with 'description' and 'price'
            **kwargs: Additional fields for output

        Returns:
            Self for chaining
        """
        output = ParserOutput(
            merchant=merchant,
            date=date,
            total=total,
            items=items or [],
            **{k: v for k, v in kwargs.items() if k in ParserOutput.__dataclass_fields__}
        )
        self._outputs.append(output)
        return self

    def set_parse_output(self, output: ParserOutput) -> "FakeReceiptParser":
        """Set full ParserOutput directly."""
        self._outputs.append(output)
        return self

    def set_default_output(
        self,
        merchant: str = "Test Store",
        total: float = 10.00,
        date: str = "2024-01-01"
    ) -> "FakeReceiptParser":
        """Set default output when no specific config provided."""
        self._default_output = ParserOutput(
            merchant=merchant,
            total=total,
            date=date,
            items=[{"description": "Test Item", "price": total}]
        )
        return self

    def set_parse_fails_with(self, exception: Exception) -> "FakeReceiptParser":
        """Configure parser to fail with given exception."""
        self._outputs.append(exception)
        return self

    def set_json_output(self, json_str: str) -> "FakeReceiptParser":
        """Set raw JSON string output (for testing JSON parsing)."""
        try:
            data = json.loads(json_str)
            output = ParserOutput(
                merchant=data.get("merchant", ""),
                date=data.get("date", ""),
                total=data.get("total"),
                items=data.get("items", []),
                raw_json=json_str
            )
            self._outputs.append(output)
        except json.JSONDecodeError as e:
            self._outputs.append(e)
        return self

    def set_token_usage(self, input_tokens: int, output_tokens: int) -> "FakeReceiptParser":
        """Configure token usage to report."""
        self._token_usage = {"input": input_tokens, "output": output_tokens}
        return self

    def set_retry_strategy_used(self, strategy: str) -> "FakeReceiptParser":
        """Record that a retry strategy was used."""
        self._retry_strategies.append(strategy)
        return self

    def set_sequence(self, sequence: list) -> "FakeReceiptParser":
        """Set a sequence of outputs for sequential calls.

        Each element can be:
        - ParserOutput for success
        - Exception for failure

        Example:
            >>> parser.set_sequence([
            ...     ValueError("Missing field"),
            ...     ParserOutput(merchant="Store", total=10.0)
            ... ])

        Args:
            sequence: List of outputs/exceptions to return in order

        Returns:
            Self for chaining
        """
        self._outputs = list(sequence)
        self._output_index = 0
        return self

    def get_attempt_history(self) -> list:
        """Get detailed history of all parse attempts.

        Returns list of dicts with:
        - attempt_number: int
        - success: bool
        - exception_type: str (if failed)
        - merchant: str (if succeeded)
        """
        return list(self._attempt_history)

    def reset_output_sequence(self) -> None:
        """Reset to first output (for retry testing)."""
        self._output_index = 0
        self._retry_strategies.clear()
        self._attempt_history.clear()

    def _get_next_output(self) -> Union[ParserOutput, Exception]:
        """Get the next configured output."""
        if self._output_index < len(self._outputs):
            output = self._outputs[self._output_index]
            self._output_index += 1
            return output

        if self._default_output:
            return self._default_output

        raise ConfigurationError(
            "No parse output configured. Use set_parse_result() or set_default_output()."
        )

    # ReceiptParsingInterface implementation

    def parse_text(self, text: str) -> "APIResponse":
        """Parse OCR text into structured receipt data.

        Returns APIResponse with parsed data or error.
        """
        from api_response import APIResponse

        start = time.time()
        output = self._get_next_output()
        attempt_number = len(self._attempt_history) + 1

        # Handle exceptions
        if isinstance(output, Exception):
            duration_ms = (time.time() - start) * 1000

            # Track attempt history
            history_entry = {
                "attempt_number": attempt_number,
                "success": False,
                "exception_type": type(output).__name__,
                "merchant": None,
            }
            self._attempt_history.append(history_entry)

            self._record_call(
                "parse_text",
                (text,),
                {},
                exception=output,
                duration_ms=duration_ms
            )
            raise output

        # Build response data
        result_data = {
            "merchant_name": output.merchant,
            "receipt_date": output.date,
            "total_amount": output.total,
            "items": output.items,
            "currency": output.currency,
            "tax_amount": output.tax,
        }

        duration_ms = (time.time() - start) * 1000

        # Track attempt history
        history_entry = {
            "attempt_number": attempt_number,
            "success": True,
            "exception_type": None,
            "merchant": output.merchant,
        }
        self._attempt_history.append(history_entry)

        self._record_call(
            "parse_text",
            (text,),
            {},
            result=result_data,
            duration_ms=duration_ms
        )

        return APIResponse.success(result_data)

    def get_token_usage(self) -> "TokenUsage":
        """Get token usage for parsing operations."""
        from tracking import TokenUsage

        usage = TokenUsage()
        usage.add_usage(
            self._token_usage.get("input", 0),
            self._token_usage.get("output", 0)
        )

        self._record_call("get_token_usage", (), {}, result=usage)
        return usage

    def get_current_retries(self) -> List[str]:
        """Get list of retry strategies used."""
        self._record_call("get_current_retries", (), {}, result=self._retry_strategies)
        return list(self._retry_strategies)

    # Helper methods for assertions

    def get_parsed_merchants(self) -> List[str]:
        """Get list of merchants from all successful parses."""
        results = []
        for call in self.get_calls("parse_text"):
            if call.result and isinstance(call.result, dict):
                results.append(call.result.get("merchant_name", ""))
        return results

    def get_parsed_totals(self) -> List[Optional[float]]:
        """Get list of totals from all successful parses."""
        results = []
        for call in self.get_calls("parse_text"):
            if call.result and isinstance(call.result, dict):
                results.append(call.result.get("total_amount"))
        return results

    def was_retried(self) -> bool:
        """Check if any retries occurred."""
        return len(self._retry_strategies) > 0

    def get_retry_count(self) -> int:
        """Get number of retries used."""
        return len(self._retry_strategies)
