"""Pure orchestrator pipeline for receipt processing"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID

from contracts.interfaces import (
    ImageProcessingInterface,
    ReceiptParsingInterface,
    BatchProcessingInterface,
    TokenUsageInterface,
    FileHandlingInterface
)
from tracking import TokenUsage
from database_models import DatabaseManager, Receipt
from storage import ReceiptRepository, UserRepository
from user_manager import UserManager
from receipt_persistence import ReceiptPersistence
from api_response import APIResponse
from image_processing import VisionProcessor
from domain.parsing.receipt_parser import ReceiptParser

from services.batch_service import BatchProcessingService
from services.token_service import TokenUsageService
from services.file_service import FileHandlingService
from core.logging import get_pipeline_logger
from core.exceptions import (
    OCRError, ParsingError, StorageError, ReceiptProcessingError
)
from config import DEFAULT_USER_EMAIL

logger = get_pipeline_logger(__name__)


class ReceiptProcessor:
    """Pure orchestrator that coordinates receipt processing through clean interfaces"""

    def __init__(self,
                 image_processor: Optional[ImageProcessingInterface] = None,
                 receipt_parser: Optional[ReceiptParsingInterface] = None,
                 db_manager: Optional[DatabaseManager] = None,
                 user_id: Optional[UUID] = None):
        # Initialize core interfaces (dependency injection with defaults)
        self.image_processor = image_processor or VisionProcessor()
        self.receipt_parser = receipt_parser or ReceiptParser()
        self.ai_parser = self.receipt_parser  # Temporary alias for test compatibility

        # Initialize token usage tracking
        self.token_usage = TokenUsage()

        # Initialize orchestration services
        self.batch_service = BatchProcessingService()
        self.token_service = TokenUsageService()
        self.file_service = FileHandlingService()

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
            # Step 1: Check for duplicate if persistence is enabled
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

            # Step 2: Create pending receipt record
            if self.repository:
                receipt_record = self.repository.create_pending_receipt(image_path, "")

            # Step 3: Extract OCR text using interface
            ocr_text = self.image_processor.extract_text(image_path)
            logger.info(f"Extracted OCR text: {ocr_text[:100]}..." if len(ocr_text) > 100 else f"Extracted OCR text: {ocr_text}")

            # Step 4: Parse with AI using interface
            parse_result = self.receipt_parser.parse_text(ocr_text)

            # Aggregate token usage from parsing result
            if parse_result.token_usage:
                self.token_usage.add_usage(
                    parse_result.token_usage.input_tokens,
                    parse_result.token_usage.output_tokens
                )

            # Step 5: Handle successful parsing
            # Note: ParsingResult uses .valid (bool), not .status (string)
            if parse_result.valid and self.repository and receipt_record:
                return self._handle_successful_parsing(parse_result, image_path, ocr_text, receipt_record)
            elif parse_result.valid:
                # No repository mode - return clean integration-level response
                parsed_data = parse_result.parsed.copy() if parse_result.parsed else {}

                # Attach token usage inside parsed_receipt
                token_usage = parse_result.token_usage
                parsed_data["_token_usage"] = {
                    "input_tokens": token_usage.input_tokens,
                    "output_tokens": token_usage.output_tokens,
                    "total_tokens": token_usage.input_tokens + token_usage.output_tokens
                }

                return APIResponse.success({
                    "image_path": image_path,
                    "ocr_text": ocr_text,
                    "parsed_receipt": parsed_data
                })
            elif not parse_result.valid and self.repository and receipt_record:
                return self._handle_failed_parsing(parse_result, receipt_record)
            elif not parse_result.valid:
                # No repository mode - convert ParsingResult failure to APIResponse
                return APIResponse.failure(parse_result.error or "Parsing failed")

        except (OCRError, ParsingError, StorageError) as e:
            # Handle specific pipeline errors
            logger.error(f"Pipeline error processing {image_path}: {str(e)}")
            return self._handle_processing_error(e, image_path, receipt_record)
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error processing {image_path}: {str(e)}")
            return self._handle_processing_error(
                ReceiptProcessingError(f"Unexpected pipeline error: {str(e)}"),
                image_path,
                receipt_record
            )

    def _handle_successful_parsing(self, parse_result, image_path: str, ocr_text: str, receipt_record) -> APIResponse:
        """Handle successful parsing with persistence"""
        # Extract token usage from ParsingResult
        token_usage = parse_result.token_usage
        input_tokens = token_usage.input_tokens
        output_tokens = token_usage.output_tokens
        estimated_cost = token_usage.get_estimated_cost()

        # Save receipt with idempotency handling
        updated_receipt, save_status = self.repository.save_receipt(
            image_path, ocr_text, parse_result.parsed, input_tokens, output_tokens, estimated_cost
        )

        if save_status == "duplicate":
            logger.info(f"Duplicate receipt detected: {updated_receipt.id}")
            return APIResponse.success({
                "receipt_id": str(updated_receipt.id),
                "status": updated_receipt.status,
                "data": updated_receipt.parsed_data,
                "duplicate": True,
                "duplicate_type": "existing"
            })

        # Enrich result with metadata
        metadata = {
            "receipt_id": str(updated_receipt.id),
            "persisted": True,
            "save_status": save_status,
            "_token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }

        return self.file_service.format_result(parse_result, Path(image_path))

    def _handle_failed_parsing(self, parse_result, receipt_record) -> APIResponse:
        """Handle failed parsing with persistence"""
        # Extract token usage from ParsingResult
        token_usage = parse_result.token_usage
        input_tokens = token_usage.input_tokens
        output_tokens = token_usage.output_tokens
        estimated_cost = token_usage.get_estimated_cost()

        updated_receipt = self.repository.update_receipt_failure(
            receipt_record.id, parse_result.error or "Unknown error",
            input_tokens, output_tokens, estimated_cost
        )

        metadata = {
            "receipt_id": str(updated_receipt.id),
            "persisted": True,
            "_token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }

        return APIResponse.failure(parse_result.error, metadata)

    def _handle_processing_error(self, error: Exception, image_path: str, receipt_record) -> APIResponse:
        """Handle processing errors with persistence"""
        if self.repository and receipt_record:
            input_tokens = self.token_usage.input_tokens
            output_tokens = self.token_usage.output_tokens
            estimated_cost = self.token_usage.get_estimated_cost()

            self.repository.update_receipt_failure(
                receipt_record.id, str(error), input_tokens, output_tokens, estimated_cost
            )

        return APIResponse.failure(f"Processing error: {str(error)}")

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


# Pure orchestration functions that delegate to services
def process_single_image(image_path: Path, processor: ReceiptProcessor) -> APIResponse:
    """Process a single image file"""
    result = processor.process_image(str(image_path))
    return processor.file_service.process_single_image_result(result, image_path)


def validate_and_get_image_files(imgs_dir: Path) -> List[Path]:
    """Validate and get list of image files to process"""
    file_service = FileHandlingService()
    return file_service.validate_and_get_image_files(imgs_dir)


def print_batch_summary(successful: int, failed: int, total: int):
    """Print batch processing summary"""
    batch_service = BatchProcessingService()
    batch_service.print_batch_summary(successful, failed, total)


def print_token_usage_summary(token_usage: TokenUsage):
    """Print token usage summary for current batch"""
    token_service = TokenUsageService()
    token_service.print_usage_summary(token_usage)


def save_token_usage_to_persistence(token_usage: TokenUsage):
    """Save token usage to persistent storage"""
    import uuid
    token_service = TokenUsageService()
    # Use a default user ID when running without database
    default_user_id = uuid.uuid4()
    token_service.save_token_usage_to_persistence(default_user_id, token_usage)


def print_usage_summary(show_persisted: bool = False):
    """Print token usage summary from persistent storage"""
    token_service = TokenUsageService()
    token_service.print_usage_summary(show_persisted)


def print_processing_result(result: APIResponse, image_files: List[Path], index: int):
    """Print the result of processing a single image"""
    file_service = FileHandlingService()
    file_service.print_processing_result(result, image_files, index)
