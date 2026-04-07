"""Example usage of receipt processing with database persistence"""

import os
from uuid import uuid4
from dotenv import load_dotenv

from image_processing import VisionProcessor
from ai_parsing import ReceiptParser
from pipeline.processor import ReceiptProcessor
from database_models import DatabaseManager
from receipt_persistence import ReceiptPersistence

# Load environment variables
load_dotenv()

def main():
    """Example of processing receipts with database persistence"""

    # Database configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/receipts_db")

    # Initialize database manager
    db_manager = DatabaseManager(DATABASE_URL)

    # Create tables (run once during setup)
    db_manager.create_tables()

    # Get or create user
    user_email = os.getenv("USER_EMAIL", "test@example.com")
    persistence = ReceiptPersistence(db_manager, uuid4())  # Use actual user_id in production
    user = persistence.get_or_create_user(user_email)

    # Initialize components
    image_processor = VisionProcessor()
    ai_parser = ReceiptParser()

    # Create workflow orchestrator with persistence
    workflow = WorkflowOrchestrator(
        image_processor=image_processor,
        ai_parser=ai_parser,
        db_manager=db_manager,
        user_id=user.id
    )

    # Process a receipt
    image_path = "path/to/receipt.jpg"
    result = workflow.process_image(image_path)

    if result.success:
        print(f"✅ Receipt processed successfully!")
        print(f"Receipt ID: {result.data.get('receipt_id')}")
        print(f"Data: {result.data}")
    else:
        print(f"❌ Processing failed: {result.error}")

    # Get user's receipts
    receipts_result = workflow.get_user_receipts(limit=10)
    if receipts_result.success:
        print(f"📋 User has {len(receipts_result.data['receipts'])} receipts")
        for receipt in receipts_result.data['receipts']:
            print(f"  - {receipt['id']}: {receipt['merchant_name']} (${receipt['total_amount']})")

    # Get token usage summary
    summary = workflow.get_token_usage_summary()
    print(f"💰 Token usage: {summary}")

    # Cleanup
    db_manager.close()

def main_legacy():
    """Example of processing receipts without persistence (backward compatibility)"""

    # Initialize components (no database)
    image_processor = VisionProcessor()
    ai_parser = ReceiptParser()

    # Create workflow orchestrator without persistence
    workflow = WorkflowOrchestrator(
        image_processor=image_processor,
        ai_parser=ai_parser
        # No db_manager or user_id = no persistence
    )

    # Process a receipt (same API, no database storage)
    image_path = "path/to/receipt.jpg"
    result = workflow.process_image(image_path)

    if result.success:
        print(f"✅ Receipt processed successfully!")
        print(f"Data: {result.data}")
    else:
        print(f"❌ Processing failed: {result.error}")

    # Get token usage summary
    summary = workflow.get_token_usage_summary()
    print(f"💰 Token usage: {summary}")

if __name__ == "__main__":
    # Choose which example to run
    main_with_persistence = True

    if main_with_persistence:
        main()
    else:
        main_legacy()
