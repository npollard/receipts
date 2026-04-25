from typing import Any, Optional

from core.hashing import calculate_image_hash


class DuplicateDetector:
    """Handle image-hash calculation and repository duplicate lookups."""

    def __init__(self, repository: Any):
        self.repository = repository

    def compute_hash(self, image_path: str) -> str:
        return calculate_image_hash(image_path)

    def find_existing(self, image_path: str, image_hash: str) -> Optional[Any]:
        if hasattr(self.repository, "find_by_image_hash"):
            return self.repository.find_by_image_hash(image_hash)
        if hasattr(self.repository, "find_by_hash"):
            return self.repository.find_by_hash(image_hash)
        if hasattr(self.repository, "find_existing_receipt_by_image_hash"):
            return self.repository.find_existing_receipt_by_image_hash(image_path)
        return None
