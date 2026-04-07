"""Migration script to convert existing database to clean schema"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.database.init_db import CleanDatabaseInitializer
from config import DATABASE_URL, DATABASE_PATH

logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Migrate existing database to clean schema"""

    def __init__(self):
        self.initializer = CleanDatabaseInitializer()
        self.database_path = self.initializer.database_path

    def migrate_to_clean_schema(self, backup_existing: bool = True) -> bool:
        """Migrate existing database to clean schema

        Args:
            backup_existing: If True, creates backup of existing database

        Returns:
            True if migration successful, False otherwise
        """
        try:
            logger.info("🔄 Starting database migration to clean schema...")

            # Check if database exists
            if not self.database_path.exists():
                logger.info("ℹ️ No existing database found, creating new clean schema")
                return self.initializer.initialize_database()

            # Create backup if requested
            if backup_existing:
                backup_path = self._create_backup()
                logger.info(f"📦 Created database backup: {backup_path}")

            # Perform migration
            success = self.initializer._migrate_existing_database()

            if success:
                logger.info("✅ Database migration completed successfully")
                self._verify_migration()
            else:
                logger.error("❌ Database migration failed")

            return success

        except Exception as e:
            logger.error(f"❌ Migration error: {e}")
            return False

    def _create_backup(self) -> Path:
        """Create backup of existing database"""
        import shutil
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"receipts_backup_{timestamp}.db"
        backup_path = self.database_path.parent / backup_name

        shutil.copy2(self.database_path, backup_path)
        return backup_path

    def _verify_migration(self) -> bool:
        """Verify that migration was successful"""
        try:
            # Verify schema
            if not self.initializer.verify_schema():
                logger.error("❌ Schema verification failed after migration")
                return False

            # Get migration statistics
            schema_info = self.initializer.get_schema_info()

            logger.info("📊 Migration Statistics:")
            logger.info(f"  - Database: {schema_info['database_path']}")
            logger.info(f"  - Tables: {list(schema_info['tables'].keys())}")
            logger.info(f"  - Indexes: {schema_info['total_indexes']}")
            logger.info(f"  - Row counts: {schema_info['row_counts']}")

            # Verify data integrity
            self._verify_data_integrity()

            return True

        except Exception as e:
            logger.error(f"❌ Migration verification failed: {e}")
            return False

    def _verify_data_integrity(self) -> None:
        """Verify data integrity after migration"""
        try:
            import sqlite3

            with sqlite3.connect(self.database_path) as conn:
                # Check foreign key constraints
                conn.execute("PRAGMA foreign_key_check")

                # Check for orphaned records
                orphaned_items_result = conn.execute(
                    """
                    SELECT COUNT(*) FROM line_items li
                    LEFT JOIN receipts r ON li.receipt_id = r.id
                    WHERE r.id IS NULL
                """
                ).fetchone()
                orphaned_items = orphaned_items_result[0] if orphaned_items_result else 0

                if orphaned_items > 0:
                    logger.warning(f"⚠️ Found {orphaned_items} orphaned line items")

                # Check for missing receipt hashes
                missing_hashes_result = conn.execute(
                    """
                    SELECT COUNT(*) FROM receipts
                    WHERE receipt_hash IS NULL OR receipt_hash = ''
                """
                ).fetchone()
                missing_hashes = missing_hashes_result[0] if missing_hashes_result else 0

                if missing_hashes > 0:
                    logger.warning(f"⚠️ Found {missing_hashes} receipts with missing hash")

                # Verify unique constraints
                duplicate_hashes = conn.execute(
                    """
                    SELECT receipt_hash, COUNT(*) as count
                    FROM receipts
                    WHERE receipt_hash IS NOT NULL
                    GROUP BY receipt_hash
                    HAVING count > 1
                """
                ).fetchall()

                if duplicate_hashes:
                    logger.warning(f"⚠️ Found {len(duplicate_hashes)} duplicate receipt hashes")

                logger.info("✅ Data integrity verification completed")

        except Exception as e:
            logger.error(f"❌ Data integrity verification failed: {e}")

    def rollback_migration(self, backup_path: Path) -> bool:
        """Rollback migration from backup"""
        try:
            import shutil

            if not backup_path.exists():
                logger.error(f"❌ Backup file not found: {backup_path}")
                return False

            logger.info(f"🔄 Rolling back migration from: {backup_path}")

            # Restore from backup
            shutil.copy2(backup_path, self.database_path)

            logger.info("✅ Migration rollback completed")
            return True

        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False

    def analyze_migration_impact(self) -> Dict[str, Any]:
        """Analyze the impact of migration on existing data"""
        try:
            import sqlite3

            analysis = {
                'database_exists': self.database_path.exists(),
                'database_size': self.database_path.stat().st_size if self.database_path.exists() else 0,
                'tables': {},
                'potential_issues': [],
                'recommendations': []
            }

            if not analysis['database_exists']:
                return analysis

            with sqlite3.connect(self.database_path) as conn:
                # Analyze existing tables
                existing_tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()

                for table_name, in existing_tables:
                    table_info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                    row_count_result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                    row_count = row_count_result[0] if row_count_result else 0

                    analysis['tables'][table_name] = {
                        'columns': len(table_info),
                        'row_count': row_count,
                        'has_primary_key': any(col[5] for col in table_info)
                    }

                # Check for potential issues
                if 'receipts' in analysis['tables']:
                    receipts_info = analysis['tables']['receipts']

                    # Check for missing required fields
                    required_fields = ['id', 'user_id', 'image_path']
                    existing_columns = [col[1] for col in conn.execute("PRAGMA table_info(receipts)").fetchall()]

                    for field in required_fields:
                        if field not in existing_columns:
                            analysis['potential_issues'].append(f"Missing required field: {field}")

                    # Check data quality
                    try:
                        null_hashes_result = conn.execute(
                            "SELECT COUNT(*) FROM receipts WHERE image_hash IS NULL"
                        ).fetchone()
                        null_hashes = null_hashes_result[0] if null_hashes_result else 0
                    except sqlite3.OperationalError:
                        # Column doesn't exist, which might mean we already have clean schema
                        null_hashes = 0

                    if null_hashes > 0:
                        analysis['potential_issues'].append(f"{null_hashes} receipts have null image_hash")
                        analysis['recommendations'].append("Consider generating hashes for existing receipts")

                # Check for receipt_items table
                if 'receipt_items' in analysis['tables'] and 'line_items' not in analysis['tables']:
                    analysis['recommendations'].append("receipt_items table will be renamed to line_items")

                # Memory usage estimate
                total_rows = sum(table['row_count'] for table in analysis['tables'].values())
                if total_rows > 10000:
                    analysis['recommendations'].append("Large dataset detected - consider running migration during off-peak hours")

            return analysis

        except Exception as e:
            logger.error(f"❌ Impact analysis failed: {e}")
            return {'error': str(e)}


def run_migration(backup: bool = True) -> bool:
    """Run the complete migration process"""
    print("=== Database Migration to Clean Schema ===")
    print(f"Database: {DATABASE_PATH}")

    # Analyze impact first
    migrator = DatabaseMigrator()
    impact = migrator.analyze_migration_impact()

    print(f"\n📊 Migration Impact Analysis:")
    print(f"  - Database exists: {impact['database_exists']}")
    print(f"  - Database size: {impact['database_size']:,} bytes")
    print(f"  - Tables: {list(impact['tables'].keys())}")

    if impact.get('potential_issues'):
        print(f"  - Potential issues: {len(impact['potential_issues'])}")
        for issue in impact['potential_issues']:
            print(f"    ⚠️ {issue}")

    if impact.get('recommendations'):
        print(f"  - Recommendations: {len(impact['recommendations'])}")
        for rec in impact['recommendations']:
            print(f"    💡 {rec}")

    # Confirm migration
    if impact['database_exists']:
        response = input("\n🤔 Proceed with migration? (y/N): ")
        if response.lower() != 'y':
            print("❌ Migration cancelled")
            return False

    # Run migration
    print("\n🚀 Starting migration...")
    success = migrator.migrate_to_clean_schema(backup_existing=backup)

    if success:
        print("✅ Migration completed successfully!")

        # Show final statistics
        schema_info = migrator.initializer.get_schema_info()
        print(f"\n📈 Final Database Statistics:")
        print(f"  - Tables: {list(schema_info['tables'].keys())}")
        print(f"  - Indexes: {schema_info['total_indexes']}")
        print(f"  - Row counts: {schema_info['row_counts']}")
    else:
        print("❌ Migration failed!")
        print("💡 Check the logs for detailed error information")

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate database to clean schema")
    parser.add_argument("--no-backup", action="store_true", help="Skip database backup")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze migration impact")

    args = parser.parse_args()

    if args.analyze_only:
        migrator = DatabaseMigrator()
        impact = migrator.analyze_migration_impact()

        print("=== Migration Impact Analysis ===")
        print(f"Database: {DATABASE_PATH}")
        print(f"Exists: {impact['database_exists']}")
        print(f"Size: {impact['database_size']:,} bytes")
        print(f"Tables: {list(impact['tables'].keys())}")

        if impact.get('potential_issues'):
            print(f"\n⚠️ Potential Issues:")
            for issue in impact['potential_issues']:
                print(f"  - {issue}")

        if impact.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in impact['recommendations']:
                print(f"  - {rec}")
    else:
        run_migration(backup=not args.no_backup)
