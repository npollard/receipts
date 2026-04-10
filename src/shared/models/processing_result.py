"""Processing result models shared across layers"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from tracking import TokenUsage


@dataclass
class ProcessingResult:
    """Result from processing a single image"""
    image_path: str
    success: bool
    parsed_data: Dict[str, Any]
    retries: List[str]
    validation_error: Optional[str]
    token_usage: TokenUsage
    processing_time_ms: float
    ocr_method: str = "unknown"
    ocr_duration_ms: float = 0.0
    ocr_attempted_methods: List[str] = None
    ocr_quality_score: float = 0.0

    def __post_init__(self):
        if self.ocr_attempted_methods is None:
            self.ocr_attempted_methods = []
