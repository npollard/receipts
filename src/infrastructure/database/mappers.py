"""Mappers between domain models and ORM entities

Provides bidirectional mapping between Pydantic domain models and
SQLAlchemy ORM entities to decouple domain logic from persistence.
"""

from datetime import date
from decimal import Decimal
from typing import Optional, List

from models.receipt import Receipt as DomainReceipt, ReceiptItem as DomainReceiptItem
from .models import Receipt as OrmReceipt, ReceiptItem as OrmReceiptItem


def domain_item_to_orm(domain_item: DomainReceiptItem, line_number: int = 1) -> OrmReceiptItem:
    """Convert domain ReceiptItem to ORM ReceiptItem.

    Args:
        domain_item: Domain model receipt item
        line_number: Line number for ordering (default 1)

    Returns:
        ORM ReceiptItem entity (not yet attached to session)
    """
    return OrmReceiptItem(
        description=domain_item.description,
        quantity=Decimal('1'),  # Default quantity
        unit_price=domain_item.price,
        total_price=domain_item.price,
        line_number=line_number,
    )


def orm_item_to_domain(orm_item: OrmReceiptItem) -> DomainReceiptItem:
    """Convert ORM ReceiptItem to domain ReceiptItem.

    Args:
        orm_item: ORM ReceiptItem entity

    Returns:
        Domain ReceiptItem model
    """
    return DomainReceiptItem(
        description=orm_item.description,
        price=Decimal(str(orm_item.unit_price)) if orm_item.unit_price else Decimal('0'),
    )


def domain_to_orm(
    domain_receipt: DomainReceipt,
    user_id: str,
    image_path: str,
    image_hash: Optional[str] = None,
) -> OrmReceipt:
    """Convert domain Receipt to ORM Receipt.

    Args:
        domain_receipt: Domain model receipt
        user_id: User ID for the receipt owner
        image_path: Path to the source image
        image_hash: Optional hash of the image file

    Returns:
        ORM Receipt entity (not yet attached to session)
    """
    # Parse date from domain model
    receipt_date = None
    if domain_receipt.date:
        try:
            receipt_date = date.fromisoformat(domain_receipt.date)
        except (ValueError, TypeError):
            receipt_date = None

    # Create ORM receipt
    orm_receipt = OrmReceipt(
        user_id=user_id,
        image_path=image_path,
        receipt_hash=image_hash,
        receipt_date=receipt_date,
        total_amount=domain_receipt.total,
        status='success',
    )

    # Convert items
    if domain_receipt.items:
        orm_receipt.items = [
            domain_item_to_orm(item, line_number=i+1)
            for i, item in enumerate(domain_receipt.items)
        ]

    return orm_receipt


def orm_to_domain(orm_receipt: OrmReceipt) -> DomainReceipt:
    """Convert ORM Receipt to domain Receipt.

    Args:
        orm_receipt: ORM Receipt entity

    Returns:
        Domain Receipt model
    """
    # Convert date to string format expected by domain model
    date_str = None
    if orm_receipt.receipt_date:
        date_str = orm_receipt.receipt_date.isoformat()

    # Convert items
    domain_items = []
    if orm_receipt.items:
        domain_items = [orm_item_to_domain(item) for item in orm_receipt.items]

    return DomainReceipt(
        date=date_str,
        items=domain_items,
        total=Decimal(str(orm_receipt.total_amount)) if orm_receipt.total_amount else None,
    )


def update_orm_from_domain(
    orm_receipt: OrmReceipt,
    domain_receipt: DomainReceipt,
) -> None:
    """Update existing ORM Receipt from domain Receipt (in-place).

    Args:
        orm_receipt: Existing ORM Receipt entity to update
        domain_receipt: Domain model with updated values
    """
    # Update date
    if domain_receipt.date:
        try:
            orm_receipt.receipt_date = date.fromisoformat(domain_receipt.date)
        except (ValueError, TypeError):
            pass

    # Update total
    if domain_receipt.total is not None:
        orm_receipt.total_amount = domain_receipt.total

    # Update items (replace existing)
    if domain_receipt.items is not None:
        # Clear existing items
        orm_receipt.items = []
        # Add new items
        orm_receipt.items = [
            domain_item_to_orm(item, line_number=i+1)
            for i, item in enumerate(domain_receipt.items)
        ]


def receipt_to_dict(receipt: OrmReceipt) -> dict:
    """Convert ORM Receipt entity to plain dictionary.

    Args:
        receipt: ORM Receipt entity

    Returns:
        Dictionary representation of receipt
    """
    return {
        "id": str(receipt.id) if receipt.id else None,
        "user_id": str(receipt.user_id) if receipt.user_id else None,
        "image_path": receipt.image_path,
        "image_hash": receipt.image_hash,
        "receipt_data_hash": receipt.receipt_data_hash,
        "status": receipt.status,
        "processing_status": receipt.processing_status,
        "receipt_date": receipt.receipt_date,
        "merchant_name": receipt.merchant_name,
        "total_amount": float(receipt.total_amount) if receipt.total_amount else None,
        "subtotal": float(receipt.subtotal) if receipt.subtotal else None,
        "tax_amount": float(receipt.tax_amount) if receipt.tax_amount else None,
        "tip_amount": float(receipt.tip_amount) if receipt.tip_amount else None,
        "raw_ocr_text": receipt.raw_ocr_text,
        "ocr_confidence": float(receipt.ocr_confidence) if receipt.ocr_confidence else None,
        "input_tokens": receipt.input_tokens,
        "output_tokens": receipt.output_tokens,
        "estimated_cost": float(receipt.estimated_cost) if receipt.estimated_cost else None,
        "processing_error": receipt.processing_error,
        "created_at": receipt.created_at,
        "updated_at": receipt.updated_at,
    }


def receipt_to_dto(receipt: OrmReceipt) -> "ReceiptDTO":
    """Convert ORM Receipt entity to ReceiptDTO.

    Args:
        receipt: ORM Receipt entity

    Returns:
        ReceiptDTO instance
    """
    from shared.models.receipt_dto import ReceiptDTO
    return ReceiptDTO(
        id=str(receipt.id) if receipt.id else "",
        user_id=str(receipt.user_id) if receipt.user_id else None,
        image_path=receipt.image_path,
        image_hash=receipt.image_hash,
        receipt_data_hash=receipt.receipt_data_hash,
        status=receipt.status,
        processing_status=receipt.processing_status,
        receipt_date=receipt.receipt_date.isoformat() if receipt.receipt_date else None,
        merchant_name=receipt.merchant_name,
        total_amount=float(receipt.total_amount) if receipt.total_amount else None,
        input_tokens=receipt.input_tokens,
        output_tokens=receipt.output_tokens,
        estimated_cost=float(receipt.estimated_cost) if receipt.estimated_cost else None,
        created_at=receipt.created_at,
        updated_at=receipt.updated_at,
    )
