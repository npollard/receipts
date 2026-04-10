"""Receipt Data Transfer Object for clean API boundaries"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class ReceiptDTO:
    id: str
    user_id: Optional[str]
    image_path: Optional[str]
    image_hash: Optional[str]
    receipt_data_hash: Optional[str]
    status: Optional[str]
    processing_status: Optional[str]
    receipt_date: Optional[str]
    merchant_name: Optional[str]
    total_amount: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    estimated_cost: Optional[float]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
