"""Unified repository layer for receipt persistence operations"""

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any, List, Union, Tuple
from uuid import UUID, uuid4

from sqlalchemy import text, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError

from config import DATABASE_URL
from core.hashing import calculate_image_hash, calculate_receipt_data_hash
from core.logging import get_storage_logger
from core.exceptions import (
    StorageError, DatabaseConnectionError, DataIntegrityError,
    IdempotencyError
)
from api_response import APIResponse
from models.receipt import Receipt as ReceiptModel, ReceiptItem as ReceiptItemModel

from .models import (
    Base, User, Receipt, ReceiptItem,
    parse_receipt_date, extract_merchant_name, get_uuid_column,
    UUID_COL_TYPE, UUID_DEFAULT
)
from .session import get_session, get_read_session
from shared.models.receipt_dto import ReceiptDTO

logger = get_storage_logger(__name__)


def handle_uuid_for_db(uuid_value: UUID) -> Union[str, UUID]:
    """Convert UUID to appropriate format for database"""
    if isinstance(uuid_value, UUID):
        if DATABASE_URL.startswith("sqlite"):
            return str(uuid_value)
        return uuid_value
    return uuid_value


class DatabaseConnection:
    """Centralized database connection management

    DEPRECATED: Use session.get_session() context manager directly.
    This class is kept for backward compatibility.
    """

    def __init__(self, database_url: Optional[str] = None):
        """Initialize - url parameter ignored, uses global session"""
        self.database_url = database_url or DATABASE_URL

    def get_session(self) -> Session:
        """DEPRECATED: Use session.get_session() context manager"""
        raise DeprecationWarning("Use session.get_session() context manager instead")

    def create_tables(self):
        """Create all database tables"""
        from .session import create_tables_for_url
        create_tables_for_url(self.database_url)

    def close(self):
        """Close database connections"""
        from .session import close_all_sessions
        close_all_sessions()

    @property
    def database_path(self) -> Optional[str]:
        """Get database file path for SQLite"""
        if self.database_url.startswith("sqlite"):
            from config import DATABASE_PATH
            return str(DATABASE_PATH)
        return None

    @property
    def session_scope(self):
        """DEPRECATED: Use session.get_session() context manager"""
        return get_session

    @property
    def transaction_scope(self):
        """DEPRECATED: Use session.get_session() context manager"""
        return get_session


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

    def get_existing_receipt_for_update(self, user_id: str, image_path: str,
                                        receipt_data: Optional[Dict[str, Any]] = None) -> Optional[Receipt]:
        """Get existing receipt if it should be updated instead of creating new"""
        # First check by image hash
        existing = self.check_existing_receipt_by_image_hash(user_id, image_path)
        if existing:
            return existing

        # Then check by data hash if receipt data provided
        if receipt_data:
            existing = self.check_existing_receipt_by_data_hash(user_id, receipt_data)
            if existing:
                return existing

        return None

    def handle_integrity_error(self, error: IntegrityError, operation: str) -> str:
        """Handle database integrity errors gracefully"""
        error_str = str(error).lower()
        error_orig = str(error)

        if 'uq_receipts_user_image_hash' in error_str:
            return "duplicate_receipt"
        elif 'uq_receipts_receipt_hash' in error_str:
            return "duplicate_receipt"
        elif 'not null' in error_str:
            return "missing_required_field"
        else:
            logger.error(f"Database integrity error during {operation}: {error_orig}")
            return "integrity_error"


class ReceiptRepository:
    """Repository for receipt database operations"""

    def __init__(self, user_id: UUID, database_url: Optional[str] = None):
        self.user_id = user_id
        self._user_id_for_db = handle_uuid_for_db(user_id)
        self.database_url = database_url

    def find_existing_receipt_by_image_hash(self, image_path: str) -> Optional[ReceiptDTO]:
        """Check if receipt already exists by image hash"""
        with get_read_session(self.database_url) as session:
            receipt_hash = calculate_image_hash(image_path)
            receipt = session.query(Receipt).filter(
                Receipt.user_id == self._user_id_for_db,
                Receipt.receipt_hash == receipt_hash
            ).first()

            if receipt:
                logger.info(f"Found duplicate receipt by image hash: {receipt.id}")
                from .mappers import receipt_to_dto
                return receipt_to_dto(receipt)
            return None

    def check_duplicate_receipt_data(self, receipt_data: Dict[str, Any]) -> Optional[ReceiptDTO]:
        """Check if receipt data already exists by data hash"""
        with get_read_session(self.database_url) as session:
            receipt_data_hash = calculate_receipt_data_hash(receipt_data)
            receipt = session.query(Receipt).filter(
                Receipt.user_id == self._user_id_for_db,
                Receipt.receipt_data_hash == receipt_data_hash
            ).first()

            if receipt:
                logger.info(f"Found duplicate receipt by data hash: {receipt.id}")
                from .mappers import receipt_to_dto
                return receipt_to_dto(receipt)
            return None

    def save_receipt(self, image_path: str, ocr_text: str, parsed_response: APIResponse,
                    input_tokens: int = 0, output_tokens: int = 0,
                    estimated_cost: Decimal = Decimal('0.000000'),
                    ocr_confidence: Optional[float] = None) -> Tuple[ReceiptDTO, str]:
        """Save a complete receipt with OCR and parsed data with idempotency"""
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

        with get_session(self.database_url) as session:
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
            receipt_model = None
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
                session.flush()  # Get the ID without committing (final commit by context)
            except IntegrityError as e:
                raise DataIntegrityError(f"Data integrity error creating receipt: {str(e)}")
            except SQLAlchemyError as e:
                raise DatabaseConnectionError(f"Database error creating receipt: {str(e)}")
            except Exception as e:
                raise StorageError(f"Unexpected error creating receipt: {str(e)}")

            # Add receipt items if successful parsing
            if parsed_response.success and receipt_model and receipt_model.items:
                try:
                    self._save_receipt_items(session, receipt.id, receipt_model.items)
                except Exception as e:
                    raise StorageError(f"Failed to save receipt items: {str(e)}")

            logger.info(f"Successfully saved receipt {receipt.id} for user {self.user_id}")
            from .mappers import receipt_to_dto
            return receipt_to_dto(receipt), "created"

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

    def create_pending_receipt(self, image_path: str, ocr_text: str, ocr_confidence: Optional[float] = None) -> ReceiptDTO:
        """Create a new receipt record in pending status"""
        with get_session(self.database_url) as session:
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
            session.flush()  # ID assigned here, commit by context manager

            logger.info(f"Created pending receipt {receipt.id} for user {self.user_id}")

            # Flush and refresh to ensure all fields are populated
            session.flush()
            session.refresh(receipt)

            from .mappers import receipt_to_dto
            return receipt_to_dto(receipt)

    def update_failure(self, receipt_id: UUID, error_message: str):
        """Update receipt status to failed with error message"""
        receipt_id_for_db = handle_uuid_for_db(receipt_id)
        with get_session(self.database_url) as session:
            receipt = session.query(Receipt).filter(Receipt.id == receipt_id_for_db).first()
            if receipt:
                receipt.status = 'failed'
                receipt.processing_error = error_message
                receipt.processing_completed_at = datetime.utcnow()
                logger.info(f"Updated receipt {receipt_id} to failed status")

    def update_receipt_success(self, receipt_id: UUID, parsed_response: APIResponse,
                               input_tokens: int = 0, output_tokens: int = 0,
                               estimated_cost: Decimal = Decimal('0.000000')) -> ReceiptDTO:
        """Update receipt with successful parsing results"""
        receipt_id_for_db = handle_uuid_for_db(receipt_id)
        with get_session(self.database_url) as session:
            receipt = session.query(Receipt).filter(
                Receipt.id == receipt_id_for_db,
                Receipt.user_id == self._user_id_for_db
            ).first()

            if not receipt:
                raise StorageError(f"Receipt {receipt_id} not found")

            # Update receipt data
            receipt.status = 'success'
            receipt.processing_completed_at = datetime.utcnow()
            receipt.input_tokens = input_tokens
            receipt.output_tokens = output_tokens
            receipt.estimated_cost = estimated_cost

            # Calculate and store data hash for idempotency
            if parsed_response.success:
                # Normalize input safely
                data = parsed_response.data
                if hasattr(data, 'model_dump'):
                    data_dict = data.model_dump()
                elif isinstance(data, dict):
                    data_dict = data
                else:
                    data_dict = {}

                # ALWAYS compute hash
                receipt.receipt_data_hash = calculate_receipt_data_hash(data_dict)
                print(f"HASH: {receipt.receipt_data_hash}")

                # Update receipt fields from parsed data
                if data:
                    if hasattr(data, 'date'):
                        receipt.receipt_date = parse_receipt_date(data.date)
                    if hasattr(data, 'total'):
                        receipt.total_amount = data.total
                    if hasattr(data, 'merchant'):
                        receipt.merchant_name = extract_merchant_name(data.merchant)

                # Ensure changes are flushed and refreshed
                session.flush()
                session.refresh(receipt)

                from .mappers import receipt_to_dto
                return receipt_to_dto(receipt)

            # If not success, still return DTO
            from .mappers import receipt_to_dto
            return receipt_to_dto(receipt)

    def fetch_user_receipts(self, limit: int = 50, offset: int = 0,
                           order_by: str = 'created_at', order_dir: str = 'desc') -> List[ReceiptDTO]:
        """Fetch receipts for current user"""
        with get_read_session(self.database_url) as session:
            query = session.query(Receipt).filter(Receipt.user_id == self._user_id_for_db)

            # Apply ordering
            order_column = getattr(Receipt, order_by, Receipt.created_at)
            if order_dir.lower() == 'desc':
                query = query.order_by(order_column.desc())
            else:
                query = query.order_by(order_column.asc())

            receipts = query.offset(offset).limit(limit).all()
            from .mappers import receipt_to_dto
            return [receipt_to_dto(r) for r in receipts]


# Module-level connection cache for helper functions
_connection_cache: Dict[str, DatabaseConnection] = {}


def with_database_session(func):
    """Decorator to provide database session"""
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        conn = get_database_connection()
        with conn.session_scope() as session:
            kwargs['session'] = session
            return func(*args, **kwargs)
    return wrapper


def with_transaction(func):
    """Decorator to provide transactional scope"""
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        conn = get_database_connection()
        with conn.transaction_scope() as session:
            kwargs['session'] = session
            return func(*args, **kwargs)
    return wrapper


def get_database_connection(database_url: Optional[str] = None) -> DatabaseConnection:
    """Get or create database connection (singleton per URL)"""
    url = database_url or DATABASE_URL
    if url not in _connection_cache:
        _connection_cache[url] = DatabaseConnection(url)
    return _connection_cache[url]


def close_all_connections():
    """Close all cached database connections"""
    for conn in _connection_cache.values():
        conn.close()
    _connection_cache.clear()
