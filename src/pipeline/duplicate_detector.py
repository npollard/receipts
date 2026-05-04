from typing import Any, Optional

from core.hashing import calculate_image_hash


class DuplicateDetector:
    """Handle image-hash calculation and repository duplicate lookups."""

    def __init__(self, repository: Any):
        self.repository = repository

    def compute_hash(self, image_path: str) -> str:
        return calculate_image_hash(image_path)

    def find_existing(self, image_path: str, image_hash: str) -> Optional[Any]:
        # Prefer explicit repository lookup methods in known order
        try:
            fn = getattr(self.repository, 'find_by_image_hash', None)
            if callable(fn):
                return fn(image_hash)
        except Exception:
            pass

        try:
            fn = getattr(self.repository, 'find_by_hash', None)
            if callable(fn):
                return fn(image_hash)
        except Exception:
            pass

        try:
            fn = getattr(self.repository, 'find_existing_receipt_by_image_hash', None)
            if callable(fn):
                return fn(image_path)
        except Exception:
            pass

        return None
