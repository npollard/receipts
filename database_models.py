"""Database models for receipt persistence"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
import hashlib
import json

from sqlalchemy import create_engine, Column, String, DateTime, Integer, Boolean, Text, ForeignKey, Index, Numeric, Date, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.sql import func
import os

Base = declarative_base()


def get_uuid_column():
    """Get UUID column type based on database"""
    # Check if we're using SQLite by checking DATABASE_URL or default
    database_url = os.getenv("DATABASE_URL", "sqlite:///receipts.db")
    if database_url.startswith("sqlite"):
        return String(36), lambda: str(uuid4())  # SQLite uses String
    else:
        return PG_UUID(as_uuid=True), uuid4  # PostgreSQL uses UUID


# Get UUID column type and default function
UUID_COL_TYPE, UUID_DEFAULT = get_uuid_column()


class User(Base):
    """User model for multi-tenant support"""
    __tablename__ = 'users'

    id = Column(UUID_COL_TYPE, primary_key=True, default=UUID_DEFAULT)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    # Relationships
    receipts = relationship("Receipt", back_populates="user", cascade="all, delete-orphan")


class Receipt(Base):
    """Receipt model for storing processed receipts"""
    __tablename__ = 'receipts'

    id = Column(UUID_COL_TYPE, primary_key=True, default=UUID_DEFAULT)
    user_id = Column(UUID_COL_TYPE, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    image_path = Column(String(500), nullable=False)
    image_hash = Column(String(64), unique=True, nullable=True)  # SHA-256 hash for deduplication
    receipt_data_hash = Column(String(64), unique=True, nullable=True)  # SHA-256 hash of receipt data for idempotency
    processing_status = Column(String(20), nullable=False, default='pending')

    # Raw OCR data
    raw_ocr_text = Column(Text)
    ocr_confidence = Column(Numeric(3, 2))  # 0.00 to 1.00

    # Parsed receipt data
    receipt_date = Column(Date)
    total_amount = Column(Numeric(10, 2))
    merchant_name = Column(String(255))
    parsed_data = Column(JSON)  # Cross-compatible JSON type

    # Processing metadata
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))
    processing_error = Column(Text)
    retry_count = Column(Integer, default=0)

    # Token usage tracking
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    estimated_cost = Column(Numeric(10, 6), default=0.000000)  # in USD

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="receipts")
    items = relationship("ReceiptItem", back_populates="receipt", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_receipts_user_date', 'user_id', 'receipt_date'),
        Index('idx_receipts_status', 'processing_status'),
        Index('idx_receipts_created_at', 'created_at'),
        Index('idx_receipts_user_created', 'user_id', 'created_at'),
        Index('idx_receipts_image_hash', 'image_hash'),
        Index('idx_receipts_data_hash', 'receipt_data_hash'),
    )


class ReceiptItem(Base):
    """Receipt line items for detailed tracking"""
    __tablename__ = 'receipt_items'

    id = Column(UUID_COL_TYPE, primary_key=True, default=UUID_DEFAULT)
    receipt_id = Column(UUID_COL_TYPE, ForeignKey('receipts.id', ondelete='CASCADE'), nullable=False)

    # Item details
    description = Column(String(255), nullable=False)
    quantity = Column(Numeric(8, 2), default=1.00)
    unit_price = Column(Numeric(10, 2), nullable=False)
    total_price = Column(Numeric(10, 2), nullable=False)

    # Category information
    category = Column(String(100))
    is_taxable = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    receipt = relationship("Receipt", back_populates="items")

    # Indexes
    __table_args__ = (
        Index('idx_receipt_items_receipt_id', 'receipt_id'),
        Index('idx_receipt_items_category', 'category'),
    )


class DatabaseManager:
    """Database session management"""

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a database session"""
        return self.SessionLocal()

    def close(self):
        """Close database connections"""
        self.engine.dispose()


def calculate_image_hash(image_path: str) -> str:
    """Calculate SHA-256 hash of image file for deduplication"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(image_path, "rb") as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except FileNotFoundError:
        # If file doesn't exist, hash the path as fallback
        return hashlib.sha256(image_path.encode()).hexdigest()


def calculate_receipt_data_hash(receipt_data: Dict[str, Any]) -> str:
    """Calculate SHA-256 hash of receipt data for idempotency"""
    # Create a normalized representation of receipt data for hashing
    normalized_data = {
        'date': receipt_data.get('date'),
        'total': receipt_data.get('total'),
        'merchant': receipt_data.get('merchant', ''),
        'items': []
    }

    # Normalize items for consistent hashing
    items = receipt_data.get('items', [])
    if items:
        for item in items:
            if isinstance(item, dict):
                normalized_item = {
                    'description': str(item.get('description', '')).lower().strip(),
                    'price': float(item.get('price', 0)),
                    'quantity': float(item.get('quantity', 1))
                }
                normalized_data['items'].append(normalized_item)

        # Sort items by description and price for consistent ordering
        normalized_data['items'].sort(key=lambda x: (x['description'], x['price']))

    # Create hash from normalized data
    data_string = json.dumps(normalized_data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data_string.encode()).hexdigest()


def parse_receipt_date(date_value: Any) -> Optional[date]:
    """Parse receipt date from various formats"""
    if date_value is None:
        return None

    if isinstance(date_value, str):
        try:
            # Try ISO format first
            return datetime.fromisoformat(date_value.replace('Z', '+00:00')).date()
        except ValueError:
            # Try other common formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    return datetime.strptime(date_value, fmt).date()
                except ValueError:
                    continue
    elif isinstance(date_value, (datetime, date)):
        return date_value if isinstance(date_value, date) else date_value.date()

    return None


def extract_merchant_name(ocr_text: str, parsed_data: Dict[str, Any]) -> Optional[str]:
    """Extract merchant name from OCR text or parsed data"""
    # Try to get from parsed data first
    if parsed_data and 'merchant' in parsed_data:
        return parsed_data['merchant'][:255] if parsed_data['merchant'] else None

    # Simple extraction from OCR text (first line before first item)
    if ocr_text:
        lines = ocr_text.strip().split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            # Skip lines that look like items or totals
            if (not any(char.isdigit() for char in line) and
                len(line) > 3 and
                not line.upper().startswith(('TOTAL', 'SUBTOTAL', 'TAX'))):
                return line[:255]

    return None
