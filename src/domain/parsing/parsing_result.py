from dataclasses import dataclass
from typing import Optional

from domain.models.receipt import Receipt
from tracking import TokenUsage


@dataclass
class ParsingResult:
    """Structured result that preserves parsed data even on validation failure."""

    parsed: Optional[Receipt] = None
    valid: bool = False
    error: Optional[str] = None
    token_usage: TokenUsage = None

    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = TokenUsage()
