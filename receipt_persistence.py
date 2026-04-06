"""Receipt persistence layer"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from uuid import UUID

from api_response import APIResponse
from database_models import (
    DatabaseManager, Receipt, ReceiptItem, User,
    calculate_image_hash, calculate_receipt_data_hash, parse_receipt_date, extract_merchant_name
)
import os

logger = logging.getLogger(__name__)


def handle_uuid_for_db(uuid_value):
    """Convert UUID to appropriate format for database"""
    if isinstance(uuid_value, UUID):
        # Check if using SQLite
        database_url = os.getenv("DATABASE_URL", "sqlite:///receipts.db")
        if database_url.startswith("sqlite"):
            return str(uuid_value)  # SQLite uses string
        return uuid_value  # PostgreSQL uses UUID
    return uuid_value


class ReceiptPersistence:
    """Persistence layer for receipt processing"""

    def __init__(self, db_manager: DatabaseManager, user_id: UUID):
        self.db_manager = db_manager
        self.user_id = user_id

    def get_or_create_user(self, email: str) -> User:
        """Get existing user or create new one"""
        session = self.db_manager.get_session()
        try:
            user = session.query(User).filter(User.email == email).first()
            if not user:
                user = User(email=email)
                session.add(user)
                session.commit()
                session.refresh(user)
                logger.info(f"Created new user: {email}")
            return user
        except Exception as e:
            session.rollback()
            logger.error(f"Error getting/creating user: {e}")
            raise
        finally:
            session.close()

    def check_duplicate_receipt(self, image_path: str) -> Optional[Receipt]:
        """Check if receipt already exists by image hash"""
        session = self.db_manager.get_session()
        try:
            image_hash = calculate_image_hash(image_path)
            user_id_for_db = handle_uuid_for_db(self.user_id)
            receipt = session.query(Receipt).filter(
                Receipt.user_id == user_id_for_db,
                Receipt.image_hash == image_hash
            ).first()

            if receipt:
                logger.info(f"Found duplicate receipt by image hash: {receipt.id}")
                return receipt
            return None
        except Exception as e:
            logger.error(f"Error checking duplicate receipt: {e}")
            raise
        finally:
            session.close()

    def check_duplicate_receipt_data(self, receipt_data: Dict[str, Any]) -> Optional[Receipt]:
        """Check if receipt data already exists by data hash"""
        session = self.db_manager.get_session()
        try:
            receipt_data_hash = calculate_receipt_data_hash(receipt_data)
            user_id_for_db = handle_uuid_for_db(self.user_id)
            receipt = session.query(Receipt).filter(
                Receipt.user_id == user_id_for_db,
                Receipt.receipt_data_hash == receipt_data_hash
            ).first()

            if receipt:
                logger.info(f"Found duplicate receipt by data hash: {receipt.id}")
                return receipt
            return None
        except Exception as e:
            logger.error(f"Error checking duplicate receipt data: {e}")
            raise
        finally:
            session.close()

    def create_pending_receipt(self, image_path: str, ocr_text: str, ocr_confidence: float = None) -> Receipt:
        """Create a new receipt record in pending status"""
        session = self.db_manager.get_session()
        try:
            image_hash = calculate_image_hash(image_path)
            user_id_for_db = handle_uuid_for_db(self.user_id)

            receipt = Receipt(
                user_id=user_id_for_db,
                image_path=image_path,
                image_hash=image_hash,
                processing_status='pending',
                raw_ocr_text=ocr_text,
                ocr_confidence=Decimal(str(ocr_confidence)) if ocr_confidence else None,
                processing_started_at=datetime.utcnow()
            )

            session.add(receipt)
            session.commit()
            session.refresh(receipt)

            logger.info(f"Created pending receipt: {receipt.id}")
            return receipt
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating pending receipt: {e}")
            raise
        finally:
            session.close()

    def update_receipt_success(self, receipt_id: UUID, parsed_response: APIResponse,
                             input_tokens: int = 0, output_tokens: int = 0,
                             estimated_cost: Decimal = Decimal('0.000000')) -> Receipt:
        """Update receipt with successful parsing results"""
        session = self.db_manager.get_session()
        try:
            receipt = session.query(Receipt).filter(
                Receipt.id == receipt_id,
                Receipt.user_id == self.user_id
            ).first()

            if not receipt:
                raise ValueError(f"Receipt not found: {receipt_id}")

            # Update receipt with parsed data
            receipt.processing_status = 'success'
            receipt.processing_completed_at = datetime.utcnow()
            receipt.input_tokens = input_tokens
            receipt.output_tokens = output_tokens
            receipt.estimated_cost = estimated_cost

            if parsed_response.success and parsed_response.data:
                data = parsed_response.data

                # Calculate and set receipt data hash for idempotency
                receipt.receipt_data_hash = calculate_receipt_data_hash(data)

                # Extract basic fields
                receipt.receipt_date = parse_receipt_date(data.get('date'))
                receipt.total_amount = Decimal(str(data.get('total', 0))) if data.get('total') else None
                receipt.merchant_name = extract_merchant_name(receipt.raw_ocr_text, data)
                receipt.parsed_data = data

                # Create receipt items
                if 'items' in data and data['items']:
                    self._create_receipt_items(session, receipt.id, data['items'])

            session.commit()
            session.refresh(receipt)

            logger.info(f"Updated receipt with success: {receipt.id}")
            return receipt
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating receipt success: {e}")
            raise
        finally:
            session.close()

    def update_receipt_failure(self, receipt_id: UUID, error_message: str,
                             input_tokens: int = 0, output_tokens: int = 0,
                             estimated_cost: Decimal = Decimal('0.000000')) -> Receipt:
        """Update receipt with failure information"""
        session = self.db_manager.get_session()
        try:
            receipt = session.query(Receipt).filter(
                Receipt.id == receipt_id,
                Receipt.user_id == self.user_id
            ).first()

            if not receipt:
                raise ValueError(f"Receipt not found: {receipt_id}")

            receipt.processing_status = 'failed'
            receipt.processing_completed_at = datetime.utcnow()
            receipt.processing_error = error_message[:1000] if error_message else None  # Limit error length
            receipt.input_tokens = input_tokens
            receipt.output_tokens = output_tokens
            receipt.estimated_cost = estimated_cost

            session.commit()
            session.refresh(receipt)

            logger.info(f"Updated receipt with failure: {receipt.id}")
            return receipt
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating receipt failure: {e}")
            raise
        finally:
            session.close()

    def _create_receipt_items(self, session: Session, receipt_id: UUID, items: list):
        """Create receipt items from parsed data"""
        for item_data in items:
            if not isinstance(item_data, dict):
                continue

            # Extract item details with defaults
            description = str(item_data.get('description', 'Unknown Item'))[:255]
            quantity = Decimal(str(item_data.get('quantity', 1)))
            unit_price = Decimal(str(item_data.get('price', 0)))
            total_price = Decimal(str(item_data.get('price', 0)))  # Default to unit price

            # Calculate total if quantity > 1 and price is unit price
            if quantity != 1 and total_price == unit_price:
                total_price = quantity * unit_price

            receipt_item = ReceiptItem(
                receipt_id=receipt_id,
                description=description,
                quantity=quantity,
                unit_price=unit_price,
                total_price=total_price,
                category=item_data.get('category')[:100] if item_data.get('category') else None
            )

            session.add(receipt_item)

    def get_user_receipts(self, limit: int = 50, offset: int = 0,
                         status: Optional[str] = None) -> list[Receipt]:
        """Get user's receipts with pagination"""
        session = self.db_manager.get_session()
        try:
            query = session.query(Receipt).filter(Receipt.user_id == self.user_id)

            if status:
                query = query.filter(Receipt.processing_status == status)

            receipts = query.order_by(Receipt.created_at.desc()).offset(offset).limit(limit).all()
            return receipts
        except Exception as e:
            logger.error(f"Error getting user receipts: {e}")
            raise
        finally:
            session.close()

    def get_receipt_with_items(self, receipt_id: UUID) -> Optional[Receipt]:
        """Get receipt with all items"""
        session = self.db_manager.get_session()
        try:
            receipt = session.query(Receipt).filter(
                Receipt.id == receipt_id,
                Receipt.user_id == self.user_id
            ).first()
            return receipt
        except Exception as e:
            logger.error(f"Error getting receipt with items: {e}")
            raise
        finally:
            session.close()
