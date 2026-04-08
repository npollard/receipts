"""Main entry point for receipt processing application"""

import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

from pipeline.processor import (
    ReceiptProcessor,
    validate_and_get_image_files,
    print_batch_summary,
    print_token_usage_summary,
    save_token_usage_to_persistence,
    print_usage_summary
)
from image_processing import VisionProcessor
from domain.parsing.receipt_parser import ReceiptParser
from services.batch_service import BatchProcessingService
from database_models import DatabaseManager
from config import DATABASE_URL, IS_TEST, app_config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, app_config.LOG_LEVEL),
    format=app_config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Process receipt images using AI')
    parser.add_argument('--usage-summary-only', action='store_true',
                       help='Show only persisted token usage summary without processing images')
    parser.add_argument('--user-email', type=str, default=None,
                       help='Specify user email for multi-user mode')
    parser.add_argument('--no-db', action='store_true',
                       help='Run without database persistence (backward compatibility)')

    args = parser.parse_args()

    # Handle usage summary only request
    if args.usage_summary_only:
        print_usage_summary(show_persisted=False)
        return

    # Initialize database manager
    db_manager = None
    if not args.no_db:
        db_manager = DatabaseManager()
        db_manager.create_tables()
        logger.info(f"Database initialized: {db_manager.database_path or DATABASE_URL}")

    # Initialize processor with database support and dependency injection
    image_processor = VisionProcessor()
    receipt_parser = ReceiptParser()
    processor = ReceiptProcessor(
        image_processor=image_processor,
        receipt_parser=receipt_parser,
        db_manager=db_manager
    )

    # Set user if specified
    if args.user_email and db_manager:
        user = processor.switch_user(args.user_email)
        logger.info(f"Processing as user: {user.email} ({user.id})")
    elif db_manager:
        user_context = processor.get_user_context()
        logger.info(f"Processing as default user: {user_context['email']} ({user_context['user_id']})")

    # Process all images in imgs directory
    imgs_dir = Path('imgs')
    image_files = validate_and_get_image_files(imgs_dir)

    if not image_files:
        return

    # Process batch using the new interface-based approach
    batch_service = BatchProcessingService()
    successful, failed, token_usage = batch_service.process_batch(
        image_files,
        image_processor,
        receipt_parser
    )

    # Print batch summary
    print_batch_summary(successful, failed, len(image_files))

    # Print current batch token usage
    print_token_usage_summary(token_usage)

    # Save token usage to persistence
    save_token_usage_to_persistence(token_usage)

    # Print persisted usage summary
    print_usage_summary(show_persisted=True)


if __name__ == "__main__":
    main()
