"""Workflow orchestrator for receipt processing pipeline"""

import logging
from typing import Optional
from uuid import UUID

from image_processing import ImageProcessor
from receipt_parser import ReceiptParser
from tracking import TokenUsage
from database_models import DatabaseManager, Receipt
from storage import ReceiptRepository, UserRepository
from user_manager import UserManager
from receipt_persistence import ReceiptPersistence
from api_response import APIResponse

from config import DEFAULT_USER_EMAIL

logger = logging.getLogger(__name__)

class WorkflowOrchestrator:
    """Orchestrates the receipt processing workflow with token tracking and persistence"""

    def __init__(self, image_processor: ImageProcessor, ai_parser: ReceiptParser,
                 db_manager: Optional[DatabaseManager] = None, user_id: Optional[UUID] = None):
        self.image_processor = image_processor
        self.ai_parser = ai_parser
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

        logger.info(f"Initialized WorkflowOrchestrator with db_manager={db_manager is not None}, user_id={user_id}")

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

    def get_token_usage_summary(self) -> str:
        """Get current token usage summary"""
        return self.token_usage.get_summary()

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
            self.persistence = ReceiptPersistence(self.db_manager, current_user_id)
            logger.info(f"Switched to user: {email} ({current_user_id})")
            return user
        return None

    def get_user_context(self) -> dict:
        """Get current user context information"""
        if self.user_manager:
            return self.user_manager.get_user_context()
        return {"user_id": None, "email": None, "is_multi_user": False}

