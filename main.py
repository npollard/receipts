"""Main entry point for receipt processing application"""

import os
import sys

# =============================================================================
# CRITICAL: Enforce thread limits BEFORE any imports that might load
# torch, numpy, or easyocr. This prevents hidden parallelism.
# =============================================================================
# Add src to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and enforce thread limits immediately
from config.runtime_config import create_config_from_env, enforce_thread_limits

# Create runtime config and enforce thread limits
_runtime_config = create_config_from_env()
enforce_thread_limits(_runtime_config)
# =============================================================================

import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

from pipeline.processor import (
    ReceiptProcessor,
    validate_and_get_image_files,
    save_token_usage_to_persistence,
    print_usage_summary
)
from image_processing import VisionProcessor
from domain.parsing.receipt_parser import ReceiptParser
from services.batch_service import BatchProcessingService
from services.token_service import TokenUsageService
from infrastructure.database import DatabaseManager
from config import DATABASE_URL, IS_TEST, app_config, get_runtime_config

# Load environment variables
load_dotenv()

# Configure centralized logging
from core.logging import setup_logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# Log the runtime configuration
logger.debug(_runtime_config.get_summary())


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
        logger.debug(f"Database initialized: {db_manager.database_path or DATABASE_URL}")

    # Initialize processor with database support and dependency injection
    image_processor = VisionProcessor()
    ocr_service = image_processor.ocr_service  # Reuse same OCR instance
    receipt_parser = ReceiptParser(ocr_service=ocr_service)
    processor = ReceiptProcessor(
        image_processor=image_processor,
        receipt_parser=receipt_parser,
        db_manager=db_manager
    )

    # Set user if specified
    if args.user_email and db_manager:
        processor.switch_user(args.user_email)
        user_context = processor.get_user_context()
        logger.info(f"Processing as user: {user_context['email']} ({user_context['user_id']})")
    elif db_manager:
        user_context = processor.get_user_context()
        logger.info(f"Processing as default user: {user_context['email']} ({user_context['user_id']})")

    # Process all images in imgs directory
    imgs_dir = Path('imgs')
    image_files = validate_and_get_image_files(imgs_dir)

    if not image_files:
        return

    # Process batch using controlled concurrency
    # BatchProcessingService is the SINGLE layer controlling parallelism
    batch_service = BatchProcessingService(runtime_config=_runtime_config)
    successful, failed, token_usage, observability = batch_service.process_batch(image_files)

    # Print clean batch summary
    if observability:
        batch_service.print_batch_summary_clean(observability)

    # Use single TokenUsageService instance for persistence and summary
    token_service = TokenUsageService()

    # Save token usage to persistence
    save_token_usage_to_persistence(token_usage, token_service=token_service)

    # Print persisted usage summary (includes current batch)
    print_usage_summary(show_persisted=True, token_service=token_service)


if __name__ == "__main__":
    main()
