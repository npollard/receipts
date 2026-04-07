"""Database migration script for idempotency constraints"""

import logging
import sys
from sqlalchemy import text

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_models import DatabaseManager
from config import DATABASE_URL

logger = logging.getLogger(__name__)


def add_idempotency_constraints(database_url: str):
    """Add unique constraints for idempotency"""
    db_manager = DatabaseManager(database_url)
    session = db_manager.get_session()

    try:
        # Check database type
        if database_url.startswith("sqlite"):
            # SQLite doesn't support adding unique constraints to existing tables
            # We need to recreate the table
            logger.info("SQLite detected: recreating receipts table with constraints")

            # Get existing data
            existing_data = session.execute(text("SELECT * FROM receipts")).fetchall()
            logger.info(f"Backed up {len(existing_data)} existing receipts")

            # Drop and recreate table
            session.execute(text("DROP TABLE IF EXISTS receipt_items"))
            session.execute(text("DROP TABLE IF EXISTS receipts"))
            session.commit()

            # Recreate tables with new constraints
            db_manager.create_tables()

            # Restore data
            if existing_data:
                # Insert data back (this will trigger any duplicate constraints)
                for row in existing_data:
                    try:
                        # Convert row to dict for insertion
                        columns = [col for col in row._fields if col != 'id']  # Skip ID to let it generate
                        values = [getattr(row, col) for col in columns]

                        insert_query = f"""
                        INSERT INTO receipts ({', '.join(columns)})
                        VALUES ({', '.join(['?' for _ in columns])})
                        """
                        session.execute(text(insert_query), values)
                    except Exception as e:
                        logger.warning(f"Skipping duplicate receipt during restore: {e}")

                session.commit()
                logger.info("Restored existing receipts with idempotency constraints")

        elif database_url.startswith("postgresql"):
            # PostgreSQL can add unique constraints directly
            logger.info("PostgreSQL detected: adding unique constraints")

            # Add unique constraints for image hash
            try:
                session.execute(text("""
                    ALTER TABLE receipts
                    ADD CONSTRAINT IF NOT EXISTS uq_receipts_user_image_hash
                    UNIQUE (user_id, image_hash)
                """))
                logger.info("Added unique constraint for user_id + image_hash")
            except Exception as e:
                logger.warning(f"Could not add image hash constraint: {e}")

            # Add unique constraints for data hash
            try:
                session.execute(text("""
                    ALTER TABLE receipts
                    ADD CONSTRAINT IF NOT EXISTS uq_receipts_user_data_hash
                    UNIQUE (user_id, receipt_data_hash)
                """))
                logger.info("Added unique constraint for user_id + receipt_data_hash")
            except Exception as e:
                logger.warning(f"Could not add data hash constraint: {e}")

            session.commit()

        else:
            logger.warning(f"Unsupported database type: {database_url}")

        logger.info("Idempotency constraints added successfully")

    except Exception as e:
        session.rollback()
        logger.error(f"Error adding idempotency constraints: {e}")
        raise
    finally:
        session.close()


def verify_constraints(database_url: str):
    """Verify that idempotency constraints are in place"""
    db_manager = DatabaseManager(database_url)
    session = db_manager.get_session()

    try:
        if database_url.startswith("postgresql"):
            # Check constraints in PostgreSQL
            result = session.execute(text("""
                SELECT conname, conkey
                FROM pg_constraint
                WHERE conrelid = 'receipts'::regclass
                AND contype = 'u'
            """)).fetchall()

            constraints = [row[0] for row in result]
            logger.info(f"Found unique constraints: {constraints}")

            expected_constraints = ['uq_receipts_user_image_hash', 'uq_receipts_user_data_hash']
            missing = [c for c in expected_constraints if c not in constraints]

            if missing:
                logger.warning(f"Missing constraints: {missing}")
                return False
            else:
                logger.info("All idempotency constraints are in place")
                return True

        elif database_url.startswith("sqlite"):
            # SQLite: Check table schema
            result = session.execute(text("""
                SELECT sql FROM sqlite_master
                WHERE type='table' AND name='receipts'
            """)).fetchone()

            if result:
                schema = result[0]
                if 'UNIQUE' in schema and 'user_id' in schema:
                    logger.info("SQLite table has unique constraints")
                    return True
                else:
                    logger.warning("SQLite table missing unique constraints")
                    return False

        return False

    except Exception as e:
        logger.error(f"Error verifying constraints: {e}")
        return False
    finally:
        session.close()


if __name__ == "__main__":
    print(f"Adding idempotency constraints to: {DATABASE_URL}")
    add_idempotency_constraints(DATABASE_URL)

    print("Verifying constraints...")
    if verify_constraints(DATABASE_URL):
        print("✅ All constraints verified successfully")
    else:
        print("❌ Constraint verification failed")
