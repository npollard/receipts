"""Clean SQLite database initializer for receipt persistence"""

import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from uuid import uuid4
from datetime import datetime, date
from decimal import Decimal

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATABASE_URL, DATABASE_PATH

logger = logging.getLogger(__name__)


class CleanDatabaseInitializer:
    """Initialize and manage clean SQLite schema for receipt persistence"""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = Path(database_path or DATABASE_PATH)
        self.schema_path = Path(__file__).parent / "clean_schema.sql"

    def initialize_database(self, force_recreate: bool = False) -> bool:
        """Initialize database with clean schema

        Args:
            force_recreate: If True, drops existing tables and recreates

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure database directory exists
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if database exists and handle migration
            if self.database_path.exists() and not force_recreate:
                return self._migrate_existing_database()

            # Load and execute schema
            schema_sql = self._load_schema()

            with self._get_connection() as conn:
                conn.executescript(schema_sql)
                conn.commit()

            logger.info(f"✅ Clean database initialized: {self.database_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False

    def _load_schema(self) -> str:
        """Load SQL schema from file"""
        try:
            with open(self.schema_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper settings"""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like row access
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        return conn

    @contextmanager
    def _get_connection_context(self):
        """Context manager for database connections"""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def _migrate_existing_database(self) -> bool:
        """Migrate existing database to clean schema"""
        logger.info("🔄 Migrating existing database...")

        try:
            with self._get_connection_context() as conn:
                # Check current schema
                existing_tables = self._get_existing_tables(conn)

                # Backup existing data if needed
                backup_data = {}
                if 'receipts' in existing_tables:
                    backup_data['receipts'] = self._backup_receipts(conn)
                if 'receipt_items' in existing_tables:
                    backup_data['receipt_items'] = self._backup_receipt_items(conn)
                if 'users' in existing_tables:
                    backup_data['users'] = self._backup_users(conn)

                # Drop existing tables
                for table in existing_tables:
                    conn.execute(f"DROP TABLE IF EXISTS {table}")

                # Create new schema
                schema_sql = self._load_schema()
                conn.executescript(schema_sql)

                # Restore data
                self._restore_data(conn, backup_data)
                conn.commit()

            logger.info("✅ Database migration completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Database migration failed: {e}")
            return False

    def _get_existing_tables(self, conn: sqlite3.Connection) -> List[str]:
        """Get list of existing tables"""
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return [row[0] for row in result]

    def _backup_receipts(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Backup existing receipts data"""
        receipts = conn.execute("SELECT * FROM receipts").fetchall()
        return [dict(row) for row in receipts]

    def _backup_receipt_items(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Backup existing receipt items data"""
        items = conn.execute("SELECT * FROM receipt_items").fetchall()
        return [dict(row) for row in items]

    def _backup_users(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Backup existing users data"""
        users = conn.execute("SELECT * FROM users").fetchall()
        return [dict(row) for row in users]

    def _restore_data(self, conn: sqlite3.Connection, backup_data: Dict[str, List[Dict[str, Any]]]):
        """Restore backed up data to new schema"""

        # Restore users first
        if 'users' in backup_data:
            for user_data in backup_data['users']:
                self._insert_user(conn, user_data)

        # Restore receipts
        if 'receipts' in backup_data:
            for receipt_data in backup_data['receipts']:
                self._insert_receipt(conn, receipt_data)

        # Restore receipt items
        if 'receipt_items' in backup_data:
            for item_data in backup_data['receipt_items']:
                self._insert_line_item(conn, item_data)

    def _insert_user(self, conn: sqlite3.Connection, user_data: Dict[str, Any]):
        """Insert user data with field mapping"""
        # Map old fields to new schema
        mapped_data = {
            'id': user_data.get('id', str(uuid4())),
            'email': user_data.get('email'),
            'created_at': user_data.get('created_at', datetime.now()),
            'updated_at': user_data.get('updated_at', datetime.now()),
            'is_active': user_data.get('is_active', True)
        }

        columns = ', '.join(mapped_data.keys())
        placeholders = ', '.join(['?' for _ in mapped_data])

        conn.execute(
            f"INSERT INTO users ({columns}) VALUES ({placeholders})",
            list(mapped_data.values())
        )

    def _insert_receipt(self, conn: sqlite3.Connection, receipt_data: Dict[str, Any]):
        """Insert receipt data with field mapping"""
        # Map old fields to new schema
        image_hash = receipt_data.get('image_hash') or receipt_data.get('receipt_data_hash')
        if not image_hash:
            # Generate a hash from image_path if no hash exists
            image_hash = f"generated_hash_{receipt_data.get('id', 'unknown')}"

        mapped_data = {
            'id': receipt_data.get('id', str(uuid4())),
            'user_id': receipt_data.get('user_id'),
            'image_path': receipt_data.get('image_path'),
            'receipt_hash': image_hash,  # Ensure we always have a hash
            'status': receipt_data.get('processing_status', 'pending'),
            'receipt_date': receipt_data.get('receipt_date'),
            'merchant_name': receipt_data.get('merchant_name'),
            'total_amount': receipt_data.get('total_amount'),
            'raw_ocr_text': receipt_data.get('raw_ocr_text'),
            'parsed_data': receipt_data.get('parsed_data'),
            'ocr_confidence': receipt_data.get('ocr_confidence'),
            'processing_started_at': receipt_data.get('processing_started_at'),
            'processing_completed_at': receipt_data.get('processing_completed_at'),
            'processing_error': receipt_data.get('processing_error'),
            'retry_count': receipt_data.get('retry_count', 0),
            'input_tokens': receipt_data.get('input_tokens', 0),
            'output_tokens': receipt_data.get('output_tokens', 0),
            'estimated_cost': receipt_data.get('estimated_cost', 0.0),
            'created_at': receipt_data.get('created_at', datetime.now()),
            'updated_at': receipt_data.get('updated_at', datetime.now())
        }

        # Remove None values for non-nullable fields
        non_nullable_fields = ['id', 'user_id', 'image_path', 'receipt_hash', 'status']
        mapped_data = {k: v for k, v in mapped_data.items()
                      if v is not None or k in non_nullable_fields}

        # Ensure non-nullable fields have values
        for field in non_nullable_fields:
            if field not in mapped_data or mapped_data[field] is None:
                if field == 'id':
                    mapped_data[field] = str(uuid4())
                elif field == 'status':
                    mapped_data[field] = 'pending'
                elif field == 'receipt_hash':
                    mapped_data[field] = f"fallback_hash_{uuid4()}"
                else:
                    mapped_data[field] = ''

        columns = ', '.join(mapped_data.keys())
        placeholders = ', '.join(['?' for _ in mapped_data])

        conn.execute(
            f"INSERT INTO receipts ({columns}) VALUES ({placeholders})",
            list(mapped_data.values())
        )

    def _insert_line_item(self, conn: sqlite3.Connection, item_data: Dict[str, Any]):
        """Insert line item data with field mapping"""
        # Map old fields to new schema
        mapped_data = {
            'id': item_data.get('id', str(uuid4())),
            'receipt_id': item_data.get('receipt_id'),
            'line_number': item_data.get('line_number', 1),
            'description': item_data.get('description'),
            'quantity': item_data.get('quantity', 1.0),
            'unit_price': item_data.get('unit_price'),
            'total_price': item_data.get('total_price'),
            'category': item_data.get('category'),
            'is_taxable': item_data.get('is_taxable', True),
            'tax_rate': item_data.get('tax_rate', 0.0),
            'created_at': item_data.get('created_at', datetime.now()),
            'updated_at': item_data.get('updated_at', datetime.now())
        }

        # Remove None values
        mapped_data = {k: v for k, v in mapped_data.items() if v is not None}

        if mapped_data:
            columns = ', '.join(mapped_data.keys())
            placeholders = ', '.join(['?' for _ in mapped_data])

            conn.execute(
                f"INSERT INTO line_items ({columns}) VALUES ({placeholders})",
                list(mapped_data.values())
            )

    def verify_schema(self) -> bool:
        """Verify that the database schema is correct"""
        try:
            with self._get_connection_context() as conn:
                # Check required tables exist
                required_tables = ['users', 'receipts', 'line_items']
                existing_tables = self._get_existing_tables(conn)

                for table in required_tables:
                    if table not in existing_tables:
                        logger.error(f"❌ Missing required table: {table}")
                        return False

                # Check required indexes exist
                required_indexes = [
                    'idx_receipts_user_id',
                    'idx_receipts_status',
                    'idx_receipts_date',
                    'idx_receipts_user_hash',
                    'idx_line_items_receipt_id'
                ]

                existing_indexes = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
                existing_index_names = {row[0] for row in existing_indexes}

                for index in required_indexes:
                    if index not in existing_index_names:
                        logger.error(f"❌ Missing required index: {index}")
                        return False

                # Check unique constraint on receipt_hash
                unique_constraint = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='index' AND name='sqlite_autoindex_receipts_1'"
                ).fetchone()

                if not unique_constraint:
                    logger.error("❌ Missing unique constraint on receipt_hash")
                    return False

                logger.info("✅ Database schema verification passed")
                return True

        except Exception as e:
            logger.error(f"❌ Schema verification failed: {e}")
            return False

    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the current database schema"""
        try:
            with self._get_connection_context() as conn:
                # Get table info
                tables = {}
                for table_name in ['users', 'receipts', 'line_items']:
                    table_info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                    tables[table_name] = {
                        'columns': [{'name': col[1], 'type': col[2], 'nullable': not col[3], 'pk': col[5]} for col in table_info],
                        'column_count': len(table_info)
                    }

                # Get index info
                indexes = conn.execute(
                    "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
                ).fetchall()

                index_info = [
                    {'name': idx[0], 'table': idx[1], 'sql': idx[2]}
                    for idx in indexes
                ]

                # Get row counts
                row_counts = {}
                for table_name in ['users', 'receipts', 'line_items']:
                    count_result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                    row_counts[table_name] = count_result[0] if count_result else 0

                return {
                    'database_path': str(self.database_path),
                    'tables': tables,
                    'indexes': index_info,
                    'row_counts': row_counts,
                    'total_indexes': len(index_info)
                }

        except Exception as e:
            logger.error(f"❌ Error getting schema info: {e}")
            return {}


# Global instance for easy access
db_initializer = CleanDatabaseInitializer()


def initialize_clean_database(force_recreate: bool = False) -> bool:
    """Initialize clean database with schema

    Args:
        force_recreate: If True, drops existing tables and recreates

    Returns:
        True if successful, False otherwise
    """
    return db_initializer.initialize_database(force_recreate)


def verify_database_schema() -> bool:
    """Verify database schema is correct"""
    return db_initializer.verify_schema()


def get_database_schema_info() -> Dict[str, Any]:
    """Get database schema information"""
    return db_initializer.get_schema_info()


if __name__ == "__main__":
    # Test database initialization
    print("=== Clean Database Initializer Test ===")
    print(f"Database path: {DATABASE_PATH}")

    success = initialize_clean_database(force_recreate=False)

    if success:
        print("✅ Database initialization successful")

        # Verify schema
        if verify_database_schema():
            print("✅ Schema verification passed")

            # Show schema info
            schema_info = get_database_schema_info()
            print(f"\nDatabase Schema Info:")
            print(f"- Path: {schema_info['database_path']}")
            print(f"- Tables: {list(schema_info['tables'].keys())}")
            print(f"- Indexes: {schema_info['total_indexes']}")
            print(f"- Row counts: {schema_info['row_counts']}")
        else:
            print("❌ Schema verification failed")
    else:
        print("❌ Database initialization failed")
