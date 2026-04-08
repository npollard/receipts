"""Clean repository layer for receipt persistence with optimized schema"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from decimal import Decimal
from datetime import datetime, date
from contextlib import contextmanager

from infrastructure.database.init_db import CleanDatabaseInitializer
from core.hashing import calculate_image_hash, calculate_data_hash
from api_response import APIResponse

logger = logging.getLogger(__name__)


class CleanReceiptRepository:
    """Repository for clean receipt persistence with optimized schema"""

    def __init__(self, database_path: Optional[str] = None, user_id: Optional[UUID] = None):
        self.initializer = CleanDatabaseInitializer(database_path)
        self.database_path = self.initializer.database_path
        self.user_id = str(user_id) if user_id else None

        # Ensure database is initialized
        if not self.initializer.verify_schema():
            self.initializer.initialize_database()

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper settings"""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def create_user(self, email: str) -> Dict[str, Any]:
        """Create a new user"""
        import uuid
        user_id = str(uuid.uuid4())
        now = datetime.now()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (id, email, created_at, updated_at, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, email, now, now, True)
            )
            conn.commit()

            return {
                'id': user_id,
                'email': email,
                'created_at': now,
                'updated_at': now,
                'is_active': True
            }

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            ).fetchone()

            return dict(row) if row else None

    def save_receipt(self, image_path: str, ocr_text: str, parsed_response: APIResponse) -> Tuple[Dict[str, Any], str]:
        """Save receipt with clean schema and idempotency"""
        import uuid

        # Calculate receipt hash for deduplication
        receipt_hash = calculate_image_hash(image_path)

        # Check for existing receipt by hash
        existing = self._get_receipt_by_hash(receipt_hash)
        if existing:
            logger.info(f"Found existing receipt by hash: {existing['id']}")
            return existing, "duplicate"

        # Create new receipt
        receipt_id = str(uuid.uuid4())
        now = datetime.now()

        receipt_data = {
            'id': receipt_id,
            'user_id': self.user_id,
            'image_path': image_path,
            'receipt_hash': receipt_hash,
            'status': 'pending',
            'raw_ocr_text': ocr_text,
            'processing_started_at': now,
            'created_at': now,
            'updated_at': now
        }

        # Add parsed data if successful
        if parsed_response.success and parsed_response.data:
            receipt_model = parsed_response.data
            receipt_data.update({
                'status': 'success',
                'receipt_date': receipt_model.date,
                'merchant_name': receipt_model.merchant_name if hasattr(receipt_model, 'merchant_name') else None,
                'total_amount': float(receipt_model.total) if receipt_model.total else None,
                'parsed_data': receipt_model.model_dump_json(),
                'processing_completed_at': now,
                'input_tokens': getattr(receipt_model, 'input_tokens', 0),
                'output_tokens': getattr(receipt_model, 'output_tokens', 0),
                'estimated_cost': float(getattr(receipt_model, 'estimated_cost', 0.0))
            })

            # Create line items
            if receipt_model.items:
                self._create_line_items(receipt_id, receipt_model.items)
        else:
            receipt_data.update({
                'status': 'failed',
                'processing_error': parsed_response.error[:1000] if parsed_response.error else None,
                'processing_completed_at': now
            })

        # Insert receipt
        with self._get_connection() as conn:
            columns = ', '.join(receipt_data.keys())
            placeholders = ', '.join(['?' for _ in receipt_data])

            conn.execute(
                f"INSERT INTO receipts ({columns}) VALUES ({placeholders})",
                list(receipt_data.values())
            )
            conn.commit()

        logger.info(f"Created receipt: {receipt_id}")
        return receipt_data, "success"

    def _get_receipt_by_hash(self, receipt_hash: str) -> Optional[Dict[str, Any]]:
        """Get receipt by hash"""
        if not self.user_id:
            return None

        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM receipts
                WHERE user_id = ? AND receipt_hash = ?
                """,
                (self.user_id, receipt_hash)
            ).fetchone()

            return dict(row) if row else None

    def _create_line_items(self, receipt_id: str, items: List[Any]) -> None:
        """Create line items for a receipt"""
        import uuid

        with self._get_connection() as conn:
            for i, item in enumerate(items, 1):
                item_id = str(uuid.uuid4())
                now = datetime.now()

                item_data = {
                    'id': item_id,
                    'receipt_id': receipt_id,
                    'line_number': i,
                    'description': item.description,
                    'quantity': float(item.quantity) if hasattr(item, 'quantity') else 1.0,
                    'unit_price': float(item.price),
                    'total_price': float(item.price) * (float(item.quantity) if hasattr(item, 'quantity') else 1.0),
                    'category': getattr(item, 'category', None),
                    'is_taxable': getattr(item, 'is_taxable', True),
                    'created_at': now,
                    'updated_at': now
                }

                columns = ', '.join(item_data.keys())
                placeholders = ', '.join(['?' for _ in item_data])

                conn.execute(
                    f"INSERT INTO line_items ({columns}) VALUES ({placeholders})",
                    list(item_data.values())
                )

    def get_receipt(self, receipt_id: str) -> Optional[Dict[str, Any]]:
        """Get receipt by ID"""
        if not self.user_id:
            return None

        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM receipts
                WHERE id = ? AND user_id = ?
                """,
                (receipt_id, self.user_id)
            ).fetchone()

            if row:
                receipt = dict(row)
                # Get line items
                receipt['line_items'] = self._get_line_items(receipt_id)
                return receipt

            return None

    def _get_line_items(self, receipt_id: str) -> List[Dict[str, Any]]:
        """Get line items for a receipt"""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM line_items
                WHERE receipt_id = ?
                ORDER BY line_number
                """,
                (receipt_id,)
            ).fetchall()

            return [dict(row) for row in rows]

    def get_user_receipts(self, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get user's receipts with pagination"""
        if not self.user_id:
            return []

        with self._get_connection() as conn:
            query = "SELECT * FROM receipts WHERE user_id = ?"
            params = [self.user_id]

            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()

            receipts = []
            for row in rows:
                receipt = dict(row)
                receipt['line_items'] = self._get_line_items(receipt['id'])
                receipts.append(receipt)

            return receipts

    def update_receipt_status(self, receipt_id: str, status: str, error: Optional[str] = None) -> bool:
        """Update receipt processing status"""
        if not self.user_id:
            return False

        with self._get_connection() as conn:
            update_data = {
                'status': status,
                'updated_at': datetime.now()
            }

            if status == 'success':
                update_data['processing_completed_at'] = datetime.now()
            elif status == 'failed' and error:
                update_data['processing_error'] = error
                update_data['processing_completed_at'] = datetime.now()

            set_clause = ', '.join([f"{k} = ?" for k in update_data.keys()])
            params = list(update_data.values()) + [receipt_id, self.user_id]

            cursor = conn.execute(
                f"""
                UPDATE receipts
                SET {set_clause}
                WHERE id = ? AND user_id = ?
                """,
                params
            )

            conn.commit()
            return cursor.rowcount > 0

    def delete_receipt(self, receipt_id: str) -> bool:
        """Delete a receipt and its line items"""
        if not self.user_id:
            return False

        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM receipts WHERE id = ? AND user_id = ?",
                (receipt_id, self.user_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_user_statistics(self) -> Dict[str, Any]:
        """Get user's receipt statistics"""
        if not self.user_id:
            return {}

        with self._get_connection() as conn:
            stats = {}

            # Total receipts
            stats['total_receipts'] = conn.execute(
                "SELECT COUNT(*) FROM receipts WHERE user_id = ?",
                (self.user_id,)
            ).scalar()

            # Status breakdown
            status_counts = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM receipts
                WHERE user_id = ?
                GROUP BY status
                """,
                (self.user_id,)
            ).fetchall()

            stats['status_breakdown'] = {row[0]: row[1] for row in status_counts}

            # Total amount and cost
            totals = conn.execute(
                """
                SELECT
                    COALESCE(SUM(total_amount), 0) as total_spent,
                    COALESCE(SUM(estimated_cost), 0) as total_cost
                FROM receipts
                WHERE user_id = ? AND status = 'success'
                """,
                (self.user_id,)
            ).fetchone()

            stats['total_spent'] = float(totals[0]) if totals[0] else 0.0
            stats['total_cost'] = float(totals[1]) if totals[1] else 0.0

            # Merchant breakdown
            merchants = conn.execute(
                """
                SELECT merchant_name, COUNT(*) as count, SUM(total_amount) as total
                FROM receipts
                WHERE user_id = ? AND status = 'success' AND merchant_name IS NOT NULL
                GROUP BY merchant_name
                ORDER BY total DESC
                LIMIT 10
                """,
                (self.user_id,)
            ).fetchall()

            stats['top_merchants'] = [
                {'merchant': row[0], 'count': row[1], 'total': float(row[2])}
                for row in merchants
            ]

            return stats

    def search_receipts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search receipts by merchant name or item descriptions"""
        if not self.user_id:
            return []

        with self._get_connection() as conn:
            # Search in receipts and line_items
            rows = conn.execute(
                """
                SELECT DISTINCT r.* FROM receipts r
                LEFT JOIN line_items li ON r.id = li.receipt_id
                WHERE r.user_id = ?
                AND (
                    r.merchant_name LIKE ?
                    OR li.description LIKE ?
                )
                ORDER BY r.created_at DESC
                LIMIT ?
                """,
                (self.user_id, f"%{query}%", f"%{query}%", limit)
            ).fetchall()

            receipts = []
            for row in rows:
                receipt = dict(row)
                receipt['line_items'] = self._get_line_items(receipt['id'])
                receipts.append(receipt)

            return receipts


# Factory function for easy instantiation
def create_clean_repository(user_id: UUID, database_path: Optional[str] = None) -> CleanReceiptRepository:
    """Create a clean repository instance"""
    return CleanReceiptRepository(database_path, user_id)
