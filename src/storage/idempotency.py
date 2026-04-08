"""Database helpers for idempotency and conflict handling"""

import logging
from typing import Optional, Dict, Any
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database_models import Receipt
from core.hashing import calculate_image_hash, calculate_data_hash

logger = logging.getLogger(__name__)


class IdempotencyHelper:
    """Helper class for handling idempotency and database conflicts"""

    def __init__(self, session: Session):
        self.session = session

    def check_existing_receipt_by_image_hash(self, user_id: str, image_path: str) -> Optional[Receipt]:
        """Check if receipt already exists by image hash"""
        receipt_hash = calculate_image_hash(image_path)

        try:
            receipt = self.session.query(Receipt).filter(
                Receipt.user_id == user_id,
                Receipt.receipt_hash == receipt_hash
            ).first()

            if receipt:
                logger.info(f"Found existing receipt by image hash: {receipt.id}")
            return receipt
        except Exception as e:
            logger.error(f"Error checking existing receipt by image hash: {e}")
            raise

    def check_existing_receipt_by_data_hash(self, user_id: str, receipt_data: Dict[str, Any]) -> Optional[Receipt]:
        """Check if receipt already exists by data hash"""
        receipt_data_hash = calculate_data_hash(receipt_data)

        try:
            receipt = self.session.query(Receipt).filter(
                Receipt.user_id == user_id,
                Receipt.receipt_data_hash == receipt_data_hash
            ).first()

            if receipt:
                logger.info(f"Found existing receipt by data hash: {receipt.id}")
            return receipt
        except Exception as e:
            logger.error(f"Error checking existing receipt by data hash: {e}")
            raise

    def handle_integrity_error(self, error: IntegrityError, operation: str) -> str:
        """Handle database integrity errors gracefully"""
        error_str = str(error).lower()
        error_orig = str(error)

        if 'uq_receipts_user_image_hash' in error_str:
            logger.warning(f"Duplicate image hash detected during {operation}")
            return "duplicate_image_hash"
        elif 'uq_receipts_user_data_hash' in error_str:
            logger.warning(f"Duplicate receipt data hash detected during {operation}")
            return "duplicate_data_hash"
        elif 'unique' in error_str:
            logger.warning(f"Unique constraint violation during {operation}: {error_orig}")
            return "unique_constraint_violation"
        else:
            logger.error(f"Integrity error during {operation}: {error_orig}")
            return "integrity_error"

    def create_receipt_with_conflict_handling(self, receipt_data: Dict[str, Any]) -> tuple[Optional[Receipt], str]:
        """Create receipt with automatic conflict handling"""
        try:
            receipt = Receipt(**receipt_data)
            self.session.add(receipt)
            self.session.flush()  # Get ID without committing
            return receipt, "success"
        except IntegrityError as e:
            self.session.rollback()
            conflict_type = self.handle_integrity_error(e, "receipt_creation")

            # Try to find the existing receipt
            if conflict_type in ["duplicate_image_hash", "duplicate_data_hash"]:
                if 'image_hash' in receipt_data:
                    existing = self.check_existing_receipt_by_image_hash(
                        receipt_data['user_id'], receipt_data['image_path']
                    )
                    if existing:
                        return existing, conflict_type

                if 'receipt_data_hash' in receipt_data:
                    existing = self.check_existing_receipt_by_data_hash(
                        receipt_data['user_id'], receipt_data.get('parsed_data', {})
                    )
                    if existing:
                        return existing, conflict_type

            return None, conflict_type

    def upsert_receipt(self, receipt_data: Dict[str, Any], update_fields: Optional[Dict[str, Any]] = None) -> tuple[Receipt, str]:
        """Upsert receipt (insert or update) with conflict handling"""
        # First try to find existing receipt
        existing = None
        if 'image_path' in receipt_data:
            existing = self.check_existing_receipt_by_image_hash(
                receipt_data['user_id'], receipt_data['image_path']
            )

        if not existing and 'parsed_data' in receipt_data:
            existing = self.check_existing_receipt_by_data_hash(
                receipt_data['user_id'], receipt_data['parsed_data']
            )

        if existing:
            # Update existing receipt
            if update_fields:
                for key, value in update_fields.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)

            self.session.flush()
            logger.info(f"Updated existing receipt: {existing.id}")
            return existing, "updated"
        else:
            # Create new receipt
            receipt, status = self.create_receipt_with_conflict_handling(receipt_data)
            if status == "success":
                logger.info(f"Created new receipt: {receipt.id}")
            return receipt, status

    def get_existing_receipt_for_update(self, user_id: str, image_path: str, parsed_data: Optional[Dict[str, Any]] = None) -> Optional[Receipt]:
        """Get existing receipt for update, checking both image and data hashes"""
        # Check by image hash first
        existing = self.check_existing_receipt_by_image_hash(user_id, image_path)
        if existing:
            return existing

        # Check by data hash if provided
        if parsed_data:
            existing = self.check_existing_receipt_by_data_hash(user_id, parsed_data)
            if existing:
                return existing

        return None
