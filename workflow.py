"""Workflow orchestration utilities"""

import logging
from typing import Dict, Any, Optional
import json
from uuid import UUID
from decimal import Decimal

from image_processing import ImageProcessor
from ai_parsing import ReceiptParser
from token_tracking import TokenUsage
from api_response import APIResponse
from models import DecimalEncoder
from database_models import DatabaseManager, Receipt
from receipt_persistence import ReceiptPersistence
from user_manager import UserManager

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Orchestrates the receipt processing workflow with token tracking and persistence"""

    def __init__(self, image_processor: ImageProcessor, ai_parser: ReceiptParser,
                 db_manager: Optional[DatabaseManager] = None, user_id: Optional[UUID] = None):
        self.image_processor = image_processor
        self.ai_parser = ai_parser
        self.token_usage = TokenUsage()
        self.db_manager = db_manager
        self.persistence = None
        self.user_manager = None

        # Initialize persistence and user management if database provided
        if db_manager:
            self.user_manager = UserManager(db_manager)

            # Use provided user_id or get default user
            if user_id:
                self.user_manager.set_user_by_id(user_id)
            else:
                self.user_manager.get_or_create_default_user()

            # Initialize persistence with current user
            current_user_id = self.user_manager.get_current_user_id()
            self.persistence = ReceiptPersistence(db_manager, current_user_id)

        logger.info("Initialized WorkflowOrchestrator")

    def process_image(self, image_path: str) -> APIResponse:
        """Process a single image through complete workflow with persistence"""
        logger.info(f"Processing image: {image_path}")

        receipt_record = None

        try:
            # Check for duplicate if persistence is enabled
            if self.persistence:
                receipt_record = self.persistence.check_duplicate_receipt(image_path)
                if receipt_record:
                    logger.info(f"Returning existing receipt: {receipt_record.id}")
                    return APIResponse.success({
                        "receipt_id": str(receipt_record.id),
                        "status": receipt_record.processing_status,
                        "data": receipt_record.parsed_data,
                        "duplicate": True
                    })

            # Create pending receipt record
            if self.persistence:
                receipt_record = self.persistence.create_pending_receipt(image_path, "")

            # Step 1: Extract OCR text
            ocr_text = self.image_processor.extract_text(image_path)
            logger.info(f"Extracted OCR text: {ocr_text[:100]}..." if len(ocr_text) > 100 else f"Extracted OCR text: {ocr_text}")

            # Update receipt with OCR data
            if self.persistence and receipt_record:
                # This would require adding update method to persistence
                pass  # OCR text is already set during creation

            # Step 2: Parse with AI (with token tracking)
            parse_result = self.ai_parser.parse_with_usage_tracking(ocr_text, self.token_usage)

            if parse_result.status == "success" and self.persistence and receipt_record:
                # Check for duplicate receipt data before saving
                duplicate_data_receipt = self.persistence.check_duplicate_receipt_data(parse_result.data)
                if duplicate_data_receipt:
                    logger.info(f"Found duplicate receipt data: {duplicate_data_receipt.id}")
                    # Update the original pending record to reference the duplicate
                    session = self.persistence.db_manager.get_session()
                    try:
                        original_receipt = session.query(Receipt).filter(Receipt.id == receipt_record.id).first()
                        if original_receipt:
                            session.delete(original_receipt)  # Remove the pending duplicate
                            session.commit()
                    except Exception as e:
                        session.rollback()
                        logger.error(f"Error removing duplicate pending record: {e}")
                    finally:
                        session.close()

                    return APIResponse.success({
                        "receipt_id": str(duplicate_data_receipt.id),
                        "status": duplicate_data_receipt.processing_status,
                        "data": duplicate_data_receipt.parsed_data,
                        "duplicate": True,
                        "duplicate_type": "data"
                    })

                # Update receipt with success
                input_tokens = self.token_usage.input_tokens
                output_tokens = self.token_usage.output_tokens
                estimated_cost = self.token_usage.get_estimated_cost()

                updated_receipt = self.persistence.update_receipt_success(
                    receipt_record.id, parse_result, input_tokens, output_tokens, estimated_cost
                )

                # Add receipt ID to response
                result_data = parse_result.data.copy() if parse_result.data else {}
                result_data["receipt_id"] = str(updated_receipt.id)
                result_data["persisted"] = True

                return APIResponse.success(result_data)

            elif parse_result.status != "success" and self.persistence and receipt_record:
                # Update receipt with failure
                input_tokens = self.token_usage.input_tokens
                output_tokens = self.token_usage.output_tokens
                estimated_cost = self.token_usage.get_estimated_cost()

                self.persistence.update_receipt_failure(
                    receipt_record.id, parse_result.error, input_tokens, output_tokens, estimated_cost
                )

            return parse_result

        except Exception as e:
            logger.error(f"Unexpected error processing {image_path}: {str(e)}")

            # Update receipt with failure if persistence enabled
            if self.persistence and receipt_record:
                input_tokens = self.token_usage.input_tokens
                output_tokens = self.token_usage.output_tokens
                estimated_cost = self.token_usage.get_estimated_cost()

                self.persistence.update_receipt_failure(
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

