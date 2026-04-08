"""Repository layer for centralized database operations"""

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any, List, Union, Tuple
from uuid import UUID, uuid4
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError

from database_models import (
    Receipt, ReceiptItem, User,
    parse_receipt_date, extract_merchant_name
)
from api_response import APIResponse
from core.hashing import calculate_image_hash, calculate_data_hash
from core.logging import get_storage_logger
from core.exceptions import (
    StorageError, DatabaseConnectionError, DataIntegrityError,
    IdempotencyError
)
from .database import DatabaseConnection, handle_uuid_for_db
from .idempotency import IdempotencyHelper
from models.receipt import Receipt as ReceiptModel, ReceiptItem as ReceiptItemModel

logger = get_storage_logger(__name__)


class ReceiptRepository:
    """Repository for receipt database operations"""

    def __init__(self, user_id: UUID, database_url: Optional[str] = None):
        self.db_connection = DatabaseConnection(database_url)
        self.user_id = user_id
        self._user_id_for_db = handle_uuid_for_db(user_id)

    def get_or_create_user(self, email: str) -> User:
        """Get existing user or create new one"""
        session = self.db_connection.get_session()
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

    def find_existing_receipt_by_image_hash(self, image_path: str) -> Optional[Receipt]:
        """Check if receipt already exists by image hash"""
        session = self.db_connection.get_session()
        try:
            receipt_hash = calculate_image_hash(image_path)
            receipt = session.query(Receipt).filter(
                Receipt.user_id == self._user_id_for_db,
                Receipt.receipt_hash == receipt_hash
            ).first()

            if receipt:
                logger.info(f"Found duplicate receipt by image hash: {receipt.id}")
            return receipt
        except Exception as e:
            logger.error(f"Error checking duplicate receipt: {e}")
            raise
        finally:
            session.close()

    def check_duplicate_receipt_data(self, receipt_data: Dict[str, Any]) -> Optional[Receipt]:
        """Check if receipt data already exists by data hash"""
        session = self.db_connection.get_session()
        try:
            receipt_data_hash = calculate_data_hash(receipt_data)
            receipt = session.query(Receipt).filter(
                Receipt.user_id == self._user_id_for_db,
                Receipt.receipt_data_hash == receipt_data_hash
            ).first()

            if receipt:
                logger.info(f"Found duplicate receipt by data hash: {receipt.id}")
            return receipt
        except Exception as e:
            logger.error(f"Error checking duplicate receipt data: {e}")
            raise
        finally:
            session.close()

    def save_receipt(self, image_path: str, ocr_text: str, parsed_response: APIResponse,
                    input_tokens: int = 0, output_tokens: int = 0,
                    estimated_cost: Decimal = Decimal('0.000000'),
                    ocr_confidence: Optional[float] = None) -> Tuple[Receipt, str]:
        """Save a complete receipt with OCR and parsed data with idempotency"""
        try:
            # Validate inputs
            if not image_path or not isinstance(image_path, str):
                raise StorageError("Invalid image path provided")

            if not ocr_text or not isinstance(ocr_text, str):
                raise StorageError("Invalid OCR text provided")

            if not isinstance(parsed_response, APIResponse):
                raise StorageError("Invalid parsed_response provided")

            # Validate that parsed_response.data is a ReceiptModel
            if parsed_response.success and parsed_response.data:
                if not isinstance(parsed_response.data, ReceiptModel):
                    raise StorageError("parsed_response.data must be a ReceiptModel instance")

            session = self.db_connection.get_session()
            try:
                idempotency_helper = IdempotencyHelper(session)

                # Check for existing receipts using idempotency helper
                try:
                    existing_receipt = idempotency_helper.get_existing_receipt_for_update(
                        self._user_id_for_db, image_path,
                        parsed_response.data.model_dump() if parsed_response.success and parsed_response.data else None
                    )
                except Exception as e:
                    raise IdempotencyError(f"Failed to check for existing receipt: {str(e)}")

                if existing_receipt:
                    logger.info(f"Returning existing receipt due to idempotency: {existing_receipt.id}")
                    return existing_receipt, "duplicate"

                # Calculate image hash
                try:
                    receipt_hash = calculate_image_hash(image_path)
                except Exception as e:
                    raise StorageError(f"Failed to calculate image hash for {image_path}: {str(e)}")

                # Prepare receipt data
                receipt_data = {
                    'user_id': self._user_id_for_db,
                    'image_path': image_path,
                    'receipt_hash': receipt_hash,
                    'status': 'success' if parsed_response.success else 'failed',
                    'raw_ocr_text': ocr_text,
                    'ocr_confidence': Decimal(str(ocr_confidence)) if ocr_confidence else None,
                    'processing_started_at': datetime.utcnow(),
                    'processing_completed_at': datetime.utcnow(),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'estimated_cost': estimated_cost
                }

                # Add parsed data if successful
                if parsed_response.success and parsed_response.data:
                    receipt_model = parsed_response.data
                    try:
                        receipt_data['receipt_date'] = parse_receipt_date(receipt_model.date)
                        receipt_data['total_amount'] = receipt_model.total
                        receipt_data['merchant_name'] = extract_merchant_name(receipt_model.merchant)
                    except Exception as e:
                        raise DataIntegrityError(f"Failed to extract receipt data: {str(e)}")

                # Create receipt record
                try:
                    receipt = Receipt(**receipt_data)
                    session.add(receipt)
                    session.flush()  # Get the ID without committing
                except IntegrityError as e:
                    raise DataIntegrityError(f"Data integrity error creating receipt: {str(e)}")
                except SQLAlchemyError as e:
                    raise DatabaseConnectionError(f"Database error creating receipt: {str(e)}")
                except Exception as e:
                    raise StorageError(f"Unexpected error creating receipt: {str(e)}")

                # Add receipt items if successful parsing
                if parsed_response.success and parsed_response.data and receipt_model.items:
                    try:
                        self._save_receipt_items(session, receipt.id, receipt_model.items)
                    except Exception as e:
                        raise StorageError(f"Failed to save receipt items: {str(e)}")

                session.commit()
                logger.info(f"Successfully saved receipt {receipt.id} for user {self.user_id}")
                return receipt, "created"

            except (StorageError, DatabaseConnectionError, DataIntegrityError, IdempotencyError):
                # Re-raise specific exceptions
                session.rollback()
                raise
            except Exception as e:
                session.rollback()
                raise StorageError(f"Unexpected error during receipt save: {str(e)}")
            finally:
                session.close()

        except (StorageError, DatabaseConnectionError, DataIntegrityError, IdempotencyError):
            # Re-raise specific exceptions
            raise
        except Exception as e:
            raise StorageError(f"Unexpected error in save_receipt: {str(e)}")

    def _save_receipt_items(self, session, receipt_id: UUID, items: List[ReceiptItemModel]):
        """Save receipt items with proper error handling"""
        try:
            for item in items:
                # Validate item data
                if not item.name:
                    raise DataIntegrityError("Receipt item name is required")

                if item.price is not None and item.price < 0:
                    raise DataIntegrityError(f"Invalid item price: {item.price}")

                item_data = {
                    'receipt_id': receipt_id,
                    'name': item.name,
                    'price': item.price,
                    'quantity': item.quantity or 1
                }

                try:
                    receipt_item = ReceiptItem(**item_data)
                    session.add(receipt_item)
                except IntegrityError as e:
                    raise DataIntegrityError(f"Data integrity error creating receipt item: {str(e)}")
                except SQLAlchemyError as e:
                    raise DatabaseConnectionError(f"Database error creating receipt item: {str(e)}")
                except Exception as e:
                    raise StorageError(f"Unexpected error creating receipt item: {str(e)}")

        except (DataIntegrityError, DatabaseConnectionError, StorageError):
            # Re-raise specific exceptions
            raise
        except Exception as e:
            raise StorageError(f"Unexpected error in _save_receipt_items: {str(e)}")

    def create_pending_receipt(self, image_path: str, ocr_text: str, ocr_confidence: Optional[float] = None) -> Receipt:
        """Create a new receipt record in pending status"""
        session = self.db_connection.get_session()
        try:
            receipt_hash = calculate_image_hash(image_path)

            receipt_data = {
                'user_id': self._user_id_for_db,
                'image_path': image_path,
                'receipt_hash': receipt_hash,
                'status': 'pending',
                'raw_ocr_text': ocr_text,
                'ocr_confidence': Decimal(str(ocr_confidence)) if ocr_confidence else None,
                'processing_started_at': datetime.utcnow(),
            }

            receipt = Receipt(**receipt_data)
            session.add(receipt)
            session.flush()  # Get the ID without committing
            session.commit()

            logger.info(f"Created pending receipt {receipt.id} for user {self.user_id}")
            return receipt

        except SQLAlchemyError as e:
            session.rollback()
            raise DatabaseConnectionError(f"Database error creating pending receipt: {str(e)}")
        except Exception as e:
            session.rollback()
            raise StorageError(f"Unexpected error creating pending receipt: {str(e)}")
        finally:
            session.close()

    def fetch_user_receipts(self, limit: int = 50, offset: int = 0,
                           status: Optional[str] = None) -> List[Receipt]:
        """Get user's receipts with pagination"""
        session = self.db_connection.get_session()
        try:
            query = session.query(Receipt).filter(Receipt.user_id == self._user_id_for_db)

            if status:
                query = query.filter(Receipt.status == status)

            receipts = query.order_by(Receipt.created_at.desc()).offset(offset).limit(limit).all()
            return receipts
        except Exception as e:
            logger.error(f"Error getting user receipts: {e}")
            raise
        finally:
            session.close()

    def get_failed_receipts(self, limit: int = 50, offset: int = 0) -> List[Receipt]:
        """Get user's failed receipts with pagination"""
        return self.get_receipts(limit=limit, offset=offset, status='failed')

    def get_receipt_with_items(self, receipt_id: UUID) -> Optional[Receipt]:
        """Get receipt with all items"""
        session = self.db_connection.get_session()
        try:
            receipt = session.query(Receipt).filter(
                Receipt.id == receipt_id,
                Receipt.user_id == self._user_id_for_db
            ).first()
            return receipt
        except Exception as e:
            logger.error(f"Error getting receipt with items: {e}")
            raise
        finally:
            session.close()

    def delete_receipt(self, receipt_id: UUID) -> bool:
        """Delete a receipt and its items"""
        session = self.db_connection.get_session()
        try:
            receipt = session.query(Receipt).filter(
                Receipt.id == receipt_id,
                Receipt.user_id == self._user_id_for_db
            ).first()

            if not receipt:
                return False

            session.delete(receipt)  # Will cascade delete items
            session.commit()
            logger.info(f"Deleted receipt: {receipt_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting receipt: {e}")
            raise
        finally:
            session.close()

    def get_receipt_statistics(self) -> Dict[str, Any]:
        """Get statistics for user's receipts"""
        session = self.db_connection.get_session()
        try:
            total_receipts = session.query(Receipt).filter(
                Receipt.user_id == self._user_id_for_db
            ).count()

            successful_receipts = session.query(Receipt).filter(
                Receipt.user_id == self._user_id_for_db,
                Receipt.status == 'success'
            ).count()

            failed_receipts = session.query(Receipt).filter(
                Receipt.user_id == self._user_id_for_db,
                Receipt.status == 'failed'
            ).count()

            total_cost = session.query(Receipt).filter(
                Receipt.user_id == self._user_id_for_db
            ).with_entities(func.coalesce(func.sum(Receipt.estimated_cost), 0)).scalar()

            return {
                'total_receipts': total_receipts,
                'successful_receipts': successful_receipts,
                'failed_receipts': failed_receipts,
                'pending_receipts': total_receipts - successful_receipts - failed_receipts,
                'total_cost': float(total_cost),
                'success_rate': (successful_receipts / total_receipts * 100) if total_receipts > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting receipt statistics: {e}")
            raise
        finally:
            session.close()

    def _create_receipt_items(self, session, receipt_id: UUID, items: List[Dict[str, Any]]):
        """Create receipt items from parsed data (legacy method for backward compatibility)"""
        # Convert dicts to Pydantic models for validation
        receipt_items = []
        for item_data in items:
            if not isinstance(item_data, dict):
                continue

            try:
                # Validate with Pydantic model
                receipt_item = ReceiptItemModel(**item_data)
                receipt_items.append(receipt_item)
            except Exception as e:
                logger.warning(f"Skipping invalid receipt item {item_data}: {e}")
                continue

        # Create items using validated models
        self._create_receipt_items_from_models(session, receipt_id, receipt_items)

    def _create_receipt_items_from_models(self, session, receipt_id: UUID, receipt_items: List[ReceiptItemModel]):
        """Create receipt items from validated Pydantic models"""
        for receipt_item in receipt_items:
            # Extract validated data from Pydantic model
            receipt_item_db = ReceiptItem(
                receipt_id=receipt_id,
                description=receipt_item.description[:255],  # Already validated
                quantity=Decimal('1'),  # Default quantity
                unit_price=receipt_item.price,  # Already validated as non-negative
                total_price=receipt_item.price,  # Same as unit price for single items
                category=None  # Not in current model
            )

            session.add(receipt_item_db)


class UserRepository:
    """Repository for user database operations"""

    def __init__(self, database_url: Optional[str] = None):
        self.db_connection = DatabaseConnection(database_url)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        session = self.db_connection.get_session()
        try:
            return session.query(User).filter(User.email == email).first()
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            raise
        finally:
            session.close()

    def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        user_id_for_db = handle_uuid_for_db(user_id)
        session = self.db_connection.get_session()
        try:
            return session.query(User).filter(User.id == user_id_for_db).first()
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            raise
        finally:
            session.close()

    def create_user(self, email: str) -> User:
        """Create a new user"""
        session = self.db_connection.get_session()
        try:
            user = User(email=email)
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info(f"Created new user: {email}")
            return user
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating user: {e}")
            raise
        finally:
            session.close()

    def get_or_create_user(self, email: str) -> User:
        """Get existing user or create new one"""
        user = self.get_user_by_email(email)
        if not user:
            user = self.create_user(email)
        return user

    def update_user(self, user_id: UUID, **kwargs) -> Optional[User]:
        """Update user attributes"""
        user_id_for_db = handle_uuid_for_db(user_id)
        session = self.db_connection.get_session()
        try:
            user = session.query(User).filter(User.id == user_id_for_db).first()
            if not user:
                return None

            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)

            session.commit()
            session.refresh(user)
            return user
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating user: {e}")
            raise
        finally:
            session.close()

    def delete_user(self, user_id: UUID) -> bool:
        """Delete a user and all their receipts"""
        user_id_for_db = handle_uuid_for_db(user_id)
        session = self.db_connection.get_session()
        try:
            user = session.query(User).filter(User.id == user_id_for_db).first()
            if not user:
                return False

            session.delete(user)  # Will cascade delete receipts
            session.commit()
            logger.info(f"Deleted user: {user_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting user: {e}")
            raise
        finally:
            session.close()
