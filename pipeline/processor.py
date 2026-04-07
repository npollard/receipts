"""Pipeline processor for receipt orchestration"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID

from image_processing import VisionProcessor
from domain.parsing.receipt_parser import ReceiptParser
from tracking import TokenUsage
from token_usage_persistence import TokenUsagePersistence
from utils.file_utils import get_image_files
from database_models import DatabaseManager, Receipt
from storage import ReceiptRepository, UserRepository
from user_manager import UserManager
from receipt_persistence import ReceiptPersistence
from api_response import APIResponse

from config import DEFAULT_USER_EMAIL

logger = logging.getLogger(__name__)


class ReceiptProcessor:
    """Unified receipt processor that coordinates all components"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None, user_id: Optional[UUID] = None):
        self.image_processor = VisionProcessor()
        self.ai_parser = ReceiptParser()
        self.token_usage = TokenUsage()

        # Initialize persistence layer
        if db_manager:
            self.user_manager = UserManager(db_manager)

            if user_id:
                self.user_manager.set_user_by_id(user_id)
            else:
                self.user_manager.get_or_create_default_user()

            current_user_id = self.user_manager.get_current_user_id()
            self.repository = ReceiptRepository(current_user_id, db_manager.engine.url)
            self.persistence = ReceiptPersistence(db_manager, current_user_id)
        else:
            self.repository = None
            self.persistence = None
            self.user_manager = None

        logger.info(f"Initialized ReceiptProcessor with db_manager={db_manager is not None}, user_id={user_id}")

    def process_image(self, image_path: str) -> APIResponse:
        """Process a single image through complete workflow with persistence"""
        logger.info(f"Processing image: {image_path}")

        receipt_record = None

        try:
            # Check for duplicate if persistence is enabled
            if self.repository:
                receipt_record = self.repository.check_existing_receipt_by_image_hash(image_path)
                if receipt_record:
                    logger.info(f"Returning existing receipt: {receipt_record.id}")
                    return APIResponse.success({
                        "receipt_id": str(receipt_record.id),
                        "status": receipt_record.status,
                        "data": receipt_record.parsed_data,
                        "duplicate": True
                    })

            # Create pending receipt record
            if self.repository:
                receipt_record = self.repository.create_pending_receipt(image_path, "")

            # Step 1: Extract OCR text
            ocr_text = self.image_processor.extract_text(image_path)
            logger.info(f"Extracted OCR text: {ocr_text[:100]}..." if len(ocr_text) > 100 else f"Extracted OCR text: {ocr_text}")

            # Step 2: Parse with AI (with token tracking)
            parse_result = self.ai_parser.parse_with_usage_tracking(ocr_text, self.token_usage)

            if parse_result.status == "success" and self.repository and receipt_record:
                # Save receipt with idempotency handling
                input_tokens = self.token_usage.input_tokens
                output_tokens = self.token_usage.output_tokens
                estimated_cost = self.token_usage.get_estimated_cost()

                updated_receipt, save_status = self.repository.save_receipt(
                    image_path, ocr_text, parse_result, input_tokens, output_tokens, estimated_cost
                )

                if save_status == "duplicate":
                    # Handle duplicate case
                    logger.info(f"Duplicate receipt detected: {updated_receipt.id}")
                    return APIResponse.success({
                        "receipt_id": str(updated_receipt.id),
                        "status": updated_receipt.status,
                        "data": updated_receipt.parsed_data,
                        "duplicate": True,
                        "duplicate_type": "existing"
                    })

                # Add receipt ID to response
                if hasattr(parse_result.data, 'model_dump'):
                    # It's a ReceiptModel - create a copy with additional fields
                    result_data = parse_result.data.model_dump()
                    result_data["receipt_id"] = str(updated_receipt.id)
                    result_data["persisted"] = True
                    result_data["save_status"] = save_status
                    # Add token usage to the result
                    result_data["_token_usage"] = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                else:
                    # It's already a dict
                    result_data = parse_result.data.copy() if parse_result.data else {}
                    result_data["receipt_id"] = str(updated_receipt.id)
                    result_data["persisted"] = True
                    result_data["save_status"] = save_status

                return APIResponse.success(result_data)
            elif parse_result.status == "success":
                # No repository mode - return result directly
                return parse_result

            elif parse_result.status != "success" and self.repository and receipt_record:
                # Update receipt with failure
                input_tokens = self.token_usage.input_tokens
                output_tokens = self.token_usage.output_tokens
                estimated_cost = self.token_usage.get_estimated_cost()

                updated_receipt = self.repository.update_receipt_failure(
                    receipt_record.id, parse_result.error or "Unknown error",
                    input_tokens, output_tokens, estimated_cost
                )

                # Add receipt ID to response
                result_data = {
                    "receipt_id": str(updated_receipt.id),
                    "persisted": True,
                    "_token_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                }

                return APIResponse.failure(parse_result.error, result_data)
            elif parse_result.status != "success":
                # No repository mode - return failure directly
                return parse_result

        except Exception as e:
            logger.error(f"Unexpected error processing {image_path}: {str(e)}")

            # Update receipt with failure if persistence enabled
            if self.repository and receipt_record:
                input_tokens = self.token_usage.input_tokens
                output_tokens = self.token_usage.output_tokens
                estimated_cost = self.token_usage.get_estimated_cost()

                self.repository.update_receipt_failure(
                    receipt_record.id, str(e), input_tokens, output_tokens, estimated_cost
                )

            return APIResponse.failure(f"Processing error: {str(e)}")

    def process_directly(self, image_path: str) -> APIResponse:
        """Process directly without LangGraph (alias for process_image)"""
        return self.process_image(image_path)

    def get_token_usage_summary(self) -> str:
        """Get current token usage summary"""
        return self.token_usage.get_summary()

    def reset_token_usage(self):
        """Reset token usage tracking"""
        self.token_usage = TokenUsage()

    def get_current_user(self):
        """Get the current user object"""
        if self.user_manager:
            return self.user_manager.get_current_user()
        return None

    def get_current_user_id(self) -> Optional[UUID]:
        """Get the current user's ID"""
        if self.user_manager:
            return self.user_manager.get_current_user_id()
        return None

    def switch_user(self, email: str):
        """Switch to a different user (creates if doesn't exist)"""
        if self.user_manager:
            user = self.user_manager.set_user_by_email(email)
            # Reinitialize persistence with new user
            current_user_id = self.user_manager.get_current_user_id()
            self.persistence = ReceiptPersistence(self.user_manager.db_manager, current_user_id)
            logger.info(f"Switched to user: {email} ({current_user_id})")
            return user
        return None

    def get_user_context(self) -> dict:
        """Get current user context information"""
        if self.user_manager:
            return self.user_manager.get_user_context()
        return {"user_id": None, "email": None, "is_multi_user": False}

    def get_user_receipts(self, limit: int = 50, offset: int = 0, status: Optional[str] = None):
        """Get user's receipts from database (if persistence enabled)"""
        if self.repository:
            return self.repository.get_user_receipts(limit, offset, status)
        return []


# Legacy function for backward compatibility
def process_receipt(image_path: str, processor: ReceiptProcessor) -> APIResponse:
    """Process a single receipt image (legacy function)"""
    return processor.process_image(image_path)


def print_processing_result(result: APIResponse, image_files: List[Path], index: int):
    """Print the result of processing a single image"""
    logger.info(f"Processing image: {image_files[index]}")

    if result.status == 'success':
        logger.info("SUCCESS")
        if result.data:
            logger.info(f"Parsed Receipt: {result.data}")
    else:
        logger.error("FAILED")
        if result.error:
            logger.error(f"Error: {result.error}")


def print_batch_summary(successful: int, failed: int, total: int):
    """Print batch processing summary"""
    logger.info("=" * 50)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(successful / total * 100):.1f}%")


def print_token_usage_summary(token_usage: TokenUsage):
    """Print token usage summary for current batch"""
    logger.info("=" * 50)
    logger.info("TOKEN USAGE SUMMARY")
    logger.info(token_usage.get_summary())
    logger.info("=" * 50)


def save_token_usage_to_persistence(token_usage: TokenUsage):
    """Save token usage to persistent storage"""
    if token_usage.get_total_tokens() > 0:
        persistence = TokenUsagePersistence()
        session_id = f"batch_session_{token_usage.get_total_tokens()}"
        persistence.save_usage(token_usage, session_id)
        logger.info(f"Saved batch token usage to persistent storage: {session_id}")


def process_batch_images(image_files: List[Path], processor: ReceiptProcessor) -> Tuple[int, int, TokenUsage]:
    """Process multiple images and return success/failure counts and token usage"""
    successful_processes = 0
    failed_processes = 0
    total_token_usage = TokenUsage()

    for i, image_path in enumerate(image_files, 1):
        result = process_receipt(str(image_path), processor)

        # Track token usage from successful results
        if result.status == 'success' and result.data:
            # Handle both dict and ReceiptModel data
            if hasattr(result.data, 'model_dump'):
                # It's a Pydantic model
                data_dict = result.data.model_dump()
            else:
                # It's already a dict
                data_dict = result.data or {}

            # Look for token usage in different possible locations
            token_usage_data = None
            if '_token_usage' in data_dict:
                token_usage_data = data_dict['_token_usage']
            elif 'parsed_receipt' in data_dict and '_token_usage' in data_dict['parsed_receipt']:
                token_usage_data = data_dict['parsed_receipt']['_token_usage']

            if token_usage_data:
                total_token_usage.add_usage(
                    token_usage_data.get('input_tokens', 0),
                    token_usage_data.get('output_tokens', 0)
                )

        # Update counters
        if result.status == 'success':
            successful_processes += 1
            logger.info("SUCCESS")
        else:
            failed_processes += 1
            logger.error("FAILED")

        # Print individual result
        print_processing_result(result, image_files, i-1)

    return successful_processes, failed_processes, total_token_usage


def process_single_image(image_path: Path, processor: ReceiptProcessor) -> APIResponse:
    """Process a single image file"""
    logger.info(f"Processing image: {image_path}")

    result = processor.process_image(str(image_path))

    if result.status == 'success':
        logger.info("Extracted Text:")
        # Handle both dict and ReceiptModel data
        if hasattr(result.data, 'model_dump'):
            # It's a Pydantic model
            data_dict = result.data.model_dump()
        else:
            # It's already a dict
            data_dict = result.data or {}

        # Print first 500 characters of extracted text
        extracted_text = data_dict.get('extracted_text', '')
        if extracted_text:
            preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
            logger.info(preview)

        logger.info("SUCCESS")

        # Print parsed receipt if available
        if 'parsed_receipt' in data_dict:
            logger.info(f"Parsed Receipt: {data_dict['parsed_receipt']}")
        elif hasattr(result.data, 'model_dump'):
            # If it's a ReceiptModel, show the model data
            logger.info(f"Parsed Receipt: {data_dict}")
    else:
        logger.error("FAILED")
        if result.error:
            logger.error(f"Error: {result.error}")

    return result


def validate_and_get_image_files(imgs_dir: Path) -> List[Path]:
    """Validate and get list of image files to process"""
    if not imgs_dir.exists():
        logger.error(f"Directory 'imgs' not found: {imgs_dir}")
        return []

    image_files = get_image_files(imgs_dir)

    if not image_files:
        logger.warning(f"No image files found in {imgs_dir}")
        return []

    logger.info(f"Found {len(image_files)} image files to process")
    return image_files


def print_usage_summary(show_persisted: bool = False):
    """Print token usage summary from persistent storage"""
    persistence = TokenUsagePersistence()
    summary = persistence.get_usage_summary()

    print("=" * 50)
    if show_persisted:
        print("PERSISTED USAGE SUMMARY")
    else:
        print("USAGE SUMMARY")
    print("=" * 50)
    print(summary)
    print("=" * 50)
