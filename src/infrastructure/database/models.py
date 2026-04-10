"""Database models for receipt persistence"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
import hashlib
import json

from sqlalchemy import create_engine, Column, String, DateTime, Integer, Boolean, Text, ForeignKey, Index, Numeric, Date, JSON
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.sql import func

from config import DATABASE_URL, DATABASE_PATH

Base = declarative_base()


def get_uuid_column():
    """Get UUID column type based on database"""
    # Check if we're using SQLite by checking DATABASE_URL
    if DATABASE_URL.startswith("sqlite"):
        return String(36), lambda: str(uuid4())  # SQLite uses String
    else:
        return PG_UUID, uuid4  # PostgreSQL uses UUID


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

    # Indexes
    __table_args__ = (
        # Email queries (most common for user lookup)
        Index('idx_users_email', 'email'),

        # Status and filtering queries
        Index('idx_users_is_active', 'is_active'),
        Index('idx_users_active_created', 'is_active', 'created_at'),

        # Time-based queries
        Index('idx_users_created_at', 'created_at'),
        Index('idx_users_updated_at', 'updated_at'),
    )


class Receipt(Base):
    """Receipt model for storing processed receipts"""
    __tablename__ = 'receipts'

    id = Column(UUID_COL_TYPE, primary_key=True, default=UUID_DEFAULT)
    user_id = Column(UUID_COL_TYPE, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    image_path = Column(String(500), nullable=False)
    image_hash = Column(String(64), index=True)  # Hash of image file for duplicate detection
    receipt_hash = Column(String(64), unique=True, nullable=True)  # SHA-256 hash for deduplication (set after processing)
    status = Column(String(20), nullable=False, default='pending')  # Processing status
    processing_status = Column(String(20), default='pending')  # Detailed processing state

    # Receipt metadata
    receipt_date = Column(Date)
    merchant_name = Column(String(255))
    total_amount = Column(Numeric(10, 2))
    subtotal = Column(Numeric(10, 2))
    tax_amount = Column(Numeric(10, 2))
    tip_amount = Column(Numeric(10, 2))

    # Raw OCR data
    raw_ocr_text = Column(Text)
    ocr_confidence = Column(Numeric(3, 2))  # 0.00 to 1.00

    # Parsed receipt data
    parsed_data = Column(Text)  # JSON as TEXT for SQLite compatibility

    # Data hash for idempotency (duplicate detection based on receipt content)
    receipt_data_hash = Column(String(64), index=True)

    # Processing metadata
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))
    processing_error = Column(Text)
    retry_count = Column(Integer, default=0)

    # Cost tracking
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    estimated_cost = Column(Numeric(10, 6), default=0.000000)  # in USD

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="receipts")
    items = relationship("ReceiptItem", back_populates="receipt", cascade="all, delete-orphan")

    # Indexes and constraints
    __table_args__ = (
        # User-scoped queries (most common pattern)
        Index('idx_receipts_user_id', 'user_id'),
        Index('idx_receipts_user_status', 'user_id', 'status'),
        Index('idx_receipts_user_created', 'user_id', 'created_at'),
        Index('idx_receipts_user_date', 'user_id', 'receipt_date'),

        # Status and filtering queries
        Index('idx_receipts_status', 'status'),
        Index('idx_receipts_status_created', 'status', 'created_at'),

        # Date/time queries (ordering and filtering)
        Index('idx_receipts_created_at', 'created_at'),
        Index('idx_receipts_receipt_date', 'receipt_date'),
        Index('idx_receipts_date_created', 'receipt_date', 'created_at'),

        # Hash-based queries (idempotency)
        Index('idx_receipts_receipt_hash', 'receipt_hash'),

        # Unique constraint on receipt_hash (already enforced by column UNIQUE)
        Index('idx_receipts_user_hash', 'user_id', 'receipt_hash', unique=True),

        # Cost and analytics queries
        Index('idx_receipts_estimated_cost', 'estimated_cost'),
        Index('idx_receipts_total_amount', 'total_amount'),
        Index('idx_receipts_user_cost', 'user_id', 'estimated_cost'),
        Index('idx_receipts_user_total', 'user_id', 'total_amount'),

        # Merchant and search queries
        Index('idx_receipts_merchant_name', 'merchant_name'),
        Index('idx_receipts_user_merchant', 'user_id', 'merchant_name'),

        # Processing and retry queries
        Index('idx_receipts_retry_count', 'retry_count'),
        Index('idx_receipts_processing_started', 'processing_started_at'),
        Index('idx_receipts_processing_completed', 'processing_completed_at'),
    )


class ReceiptItem(Base):
    """Receipt line items for detailed tracking"""
    __tablename__ = 'line_items'

    id = Column(UUID_COL_TYPE, primary_key=True, default=UUID_DEFAULT)
    receipt_id = Column(UUID_COL_TYPE, ForeignKey('receipts.id', ondelete='CASCADE'), nullable=False)

    # Item details
    line_number = Column(Integer, nullable=False, default=1)
    description = Column(String(255), nullable=False)
    quantity = Column(Numeric(8, 2), default=1.00)
    unit_price = Column(Numeric(10, 2), nullable=False)
    total_price = Column(Numeric(10, 2), nullable=False)

    # Category information
    category = Column(String(100))
    is_taxable = Column(Boolean, default=True)
    tax_rate = Column(Numeric(5, 4), default=0.0000)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    receipt = relationship("Receipt", back_populates="items")

    # Indexes
    __table_args__ = (
        # Foreign key queries (most common)
        Index('idx_receipt_items_receipt_id', 'receipt_id'),

        # Category and filtering queries
        Index('idx_receipt_items_category', 'category'),
        Index('idx_receipt_items_receipt_category', 'receipt_id', 'category'),

        # Price and analytics queries
        Index('idx_receipt_items_unit_price', 'unit_price'),
        Index('idx_receipt_items_total_price', 'total_price'),
        Index('idx_receipt_items_quantity', 'quantity'),

        # Tax and business logic queries
        Index('idx_receipt_items_is_taxable', 'is_taxable'),
        Index('idx_receipt_items_taxable_category', 'is_taxable', 'category'),

        # Search and description queries
        Index('idx_receipt_items_description', 'description'),
        Index('idx_receipt_items_receipt_description', 'receipt_id', 'description'),

        # Composite indexes for common patterns
        Index('idx_receipt_items_category_price', 'category', 'unit_price'),
        Index('idx_receipt_items_receipt_category_price', 'receipt_id', 'category', 'unit_price'),

        # Time-based queries
        Index('idx_receipt_items_created_at', 'created_at'),
        Index('idx_receipt_items_receipt_created', 'receipt_id', 'created_at'),
    )


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
