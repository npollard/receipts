"""Database migration script for performance indexes"""

import logging
import os
import sys
from sqlalchemy import text, Index
from sqlalchemy.engine import Engine

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_models import DatabaseManager
from config import DATABASE_URL

logger = logging.getLogger(__name__)


def create_performance_indexes(database_url: str):
    """Create performance indexes for improved query performance"""
    db_manager = DatabaseManager(database_url)
    engine = db_manager.engine
    session = db_manager.get_session()

    try:
        logger.info(f"Creating performance indexes for: {database_url}")

        # Define all indexes to create
        indexes_to_create = [
            # Receipt table indexes
            ('idx_receipts_user_id', 'receipts', ['user_id']),
            ('idx_receipts_user_status', 'receipts', ['user_id', 'processing_status']),
            ('idx_receipts_user_created', 'receipts', ['user_id', 'created_at']),
            ('idx_receipts_user_date', 'receipts', ['user_id', 'receipt_date']),
            ('idx_receipts_status', 'receipts', ['processing_status']),
            ('idx_receipts_status_created', 'receipts', ['processing_status', 'created_at']),
            ('idx_receipts_created_at', 'receipts', ['created_at']),
            ('idx_receipts_receipt_date', 'receipts', ['receipt_date']),
            ('idx_receipts_date_created', 'receipts', ['receipt_date', 'created_at']),
            ('idx_receipts_image_hash', 'receipts', ['image_hash']),
            ('idx_receipts_data_hash', 'receipts', ['receipt_data_hash']),
            ('idx_receipts_estimated_cost', 'receipts', ['estimated_cost']),
            ('idx_receipts_total_amount', 'receipts', ['total_amount']),
            ('idx_receipts_user_cost', 'receipts', ['user_id', 'estimated_cost']),
            ('idx_receipts_user_total', 'receipts', ['user_id', 'total_amount']),
            ('idx_receipts_merchant_name', 'receipts', ['merchant_name']),
            ('idx_receipts_user_merchant', 'receipts', ['user_id', 'merchant_name']),
            ('idx_receipts_retry_count', 'receipts', ['retry_count']),
            ('idx_receipts_processing_started', 'receipts', ['processing_started_at']),
            ('idx_receipts_processing_completed', 'receipts', ['processing_completed_at']),

            # ReceiptItem table indexes
            ('idx_receipt_items_receipt_id', 'receipt_items', ['receipt_id']),
            ('idx_receipt_items_category', 'receipt_items', ['category']),
            ('idx_receipt_items_receipt_category', 'receipt_items', ['receipt_id', 'category']),
            ('idx_receipt_items_unit_price', 'receipt_items', ['unit_price']),
            ('idx_receipt_items_total_price', 'receipt_items', ['total_price']),
            ('idx_receipt_items_quantity', 'receipt_items', ['quantity']),
            ('idx_receipt_items_is_taxable', 'receipt_items', ['is_taxable']),
            ('idx_receipt_items_taxable_category', 'receipt_items', ['is_taxable', 'category']),
            ('idx_receipt_items_description', 'receipt_items', ['description']),
            ('idx_receipt_items_receipt_description', 'receipt_items', ['receipt_id', 'description']),
            ('idx_receipt_items_category_price', 'receipt_items', ['category', 'unit_price']),
            ('idx_receipt_items_receipt_category_price', 'receipt_items', ['receipt_id', 'category', 'unit_price']),
            ('idx_receipt_items_created_at', 'receipt_items', ['created_at']),
            ('idx_receipt_items_receipt_created', 'receipt_items', ['receipt_id', 'created_at']),

            # User table indexes
            ('idx_users_email', 'users', ['email']),
            ('idx_users_is_active', 'users', ['is_active']),
            ('idx_users_active_created', 'users', ['is_active', 'created_at']),
            ('idx_users_created_at', 'users', ['created_at']),
            ('idx_users_updated_at', 'users', ['updated_at']),
        ]

        # Create indexes one by one with error handling
        created_indexes = []
        skipped_indexes = []

        for index_name, table_name, columns in indexes_to_create:
            try:
                # Check if index already exists
                result = session.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name=:index_name"
                    if database_url.startswith("sqlite")
                    else "SELECT indexname FROM pg_indexes WHERE indexname=:index_name"
                ), {"index_name": index_name})

                if result.fetchone():
                    logger.info(f"Index {index_name} already exists, skipping")
                    skipped_indexes.append(index_name)
                    continue

                # Create index
                columns_str = ", ".join(columns)
                create_sql = f"CREATE INDEX {index_name} ON {table_name} ({columns_str})"

                logger.info(f"Creating index: {index_name}")
                session.execute(text(create_sql))
                session.commit()
                created_indexes.append(index_name)
                logger.info(f"✅ Created index: {index_name}")

            except Exception as e:
                logger.error(f"❌ Error creating index {index_name}: {e}")
                session.rollback()
                continue

        logger.info(f"\nIndex Creation Summary:")
        logger.info(f"✅ Created: {len(created_indexes)} indexes")
        logger.info(f"⏭️  Skipped (already exist): {len(skipped_indexes)} indexes")

        if created_indexes:
            logger.info(f"Created indexes: {', '.join(created_indexes)}")

        return created_indexes, skipped_indexes

    except Exception as e:
        logger.error(f"Error creating performance indexes: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def analyze_query_performance(database_url: str):
    """Analyze current query performance and suggest optimizations"""
    db_manager = DatabaseManager(database_url)
    session = db_manager.get_session()

    try:
        logger.info("Analyzing query performance...")

        # Get table statistics
        if database_url.startswith("sqlite"):
            # SQLite analysis
            tables = session.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )).fetchall()

            for table in tables:
                table_name = table[0]
                count = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                logger.info(f"Table {table_name}: {count} rows")

                # Get index information
                indexes = session.execute(text(
                    "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=:table_name"
                ), {"table_name": table_name}).fetchall()

                for index in indexes:
                    logger.info(f"  Index: {index[0]} - {index[1] or '(implicit)'}")

        elif database_url.startswith("postgresql"):
            # PostgreSQL analysis
            tables = session.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            )).fetchall()

            for table in tables:
                table_name = table[0]
                count = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                logger.info(f"Table {table_name}: {count} rows")

                # Get index information
                indexes = session.execute(text(
                    "SELECT indexname, indexdef FROM pg_indexes WHERE tablename=:table_name"
                ), {"table_name": table_name}).fetchall()

                for index in indexes:
                    logger.info(f"  Index: {index[0]} - {index[1]}")

    except Exception as e:
        logger.error(f"Error analyzing performance: {e}")
    finally:
        session.close()


def verify_indexes(database_url: str):
    """Verify that all expected indexes exist"""
    db_manager = DatabaseManager(database_url)
    session = db_manager.get_session()

    try:
        logger.info("Verifying indexes...")

        # Expected indexes
        expected_indexes = {
            'receipts': [
                'idx_receipts_user_id', 'idx_receipts_user_status', 'idx_receipts_user_created',
                'idx_receipts_user_date', 'idx_receipts_status', 'idx_receipts_status_created',
                'idx_receipts_created_at', 'idx_receipts_receipt_date', 'idx_receipts_date_created',
                'idx_receipts_image_hash', 'idx_receipts_data_hash', 'idx_receipts_estimated_cost',
                'idx_receipts_total_amount', 'idx_receipts_user_cost', 'idx_receipts_user_total',
                'idx_receipts_merchant_name', 'idx_receipts_user_merchant', 'idx_receipts_retry_count',
                'idx_receipts_processing_started', 'idx_receipts_processing_completed',
                'uq_receipts_user_image_hash', 'uq_receipts_user_data_hash'
            ],
            'receipt_items': [
                'idx_receipt_items_receipt_id', 'idx_receipt_items_category',
                'idx_receipt_items_receipt_category', 'idx_receipt_items_unit_price',
                'idx_receipt_items_total_price', 'idx_receipt_items_quantity',
                'idx_receipt_items_is_taxable', 'idx_receipt_items_taxable_category',
                'idx_receipt_items_description', 'idx_receipt_items_receipt_description',
                'idx_receipt_items_category_price', 'idx_receipt_items_receipt_category_price',
                'idx_receipt_items_created_at', 'idx_receipt_items_receipt_created'
            ],
            'users': [
                'idx_users_email', 'idx_users_is_active', 'idx_users_active_created',
                'idx_users_created_at', 'idx_users_updated_at'
            ]
        }

        all_verified = True

        for table_name, index_names in expected_indexes.items():
            logger.info(f"\nVerifying {table_name} indexes:")

            for index_name in index_names:
                if database_url.startswith("sqlite"):
                    result = session.execute(text(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name=:index_name"
                    ), {"index_name": index_name})
                else:
                    result = session.execute(text(
                        "SELECT indexname FROM pg_indexes WHERE indexname=:index_name"
                    ), {"index_name": index_name})

                exists = result.fetchone() is not None
                status = "✅" if exists else "❌"
                logger.info(f"  {status} {index_name}")

                if not exists:
                    all_verified = False

        return all_verified

    except Exception as e:
        logger.error(f"Error verifying indexes: {e}")
        return False
    finally:
        session.close()


if __name__ == "__main__":
    import os

    print(f"Creating performance indexes for: {DATABASE_URL}")
    created, skipped = create_performance_indexes(DATABASE_URL)

    print(f"\nCreated {len(created)} indexes, skipped {len(skipped)} existing indexes")

    print("\nAnalyzing query performance...")
    analyze_query_performance(DATABASE_URL)

    print("\nVerifying indexes...")
    if verify_indexes(DATABASE_URL):
        print("✅ All performance indexes verified successfully")
    else:
        print("❌ Some indexes are missing")
