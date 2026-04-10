"""User Data Transfer Object for clean API boundaries"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class UserDTO:
    id: str
    email: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True
