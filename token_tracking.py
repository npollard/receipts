"""Token usage tracking utilities"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenUsage:
    """Track token usage across processing sessions"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    session_start: datetime = field(default_factory=datetime.now)

    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage from a single request"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.requests += 1

    def get_summary(self) -> str:
        """Get formatted summary of token usage"""
        duration = datetime.now() - self.session_start
        return f"""
Token Usage Summary:
- Input Tokens: {self.input_tokens:,}
- Output Tokens: {self.output_tokens:,}
- Total Tokens: {self.total_tokens:,}
- Requests: {self.requests}
- Session Duration: {duration}
- Est. Cost (GPT-4o-mini): ${(self.input_tokens * 0.00015 + self.output_tokens * 0.0006):.4f}
"""
