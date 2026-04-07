"""Pipeline processor for receipt orchestration"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from receipt_processor import ReceiptProcessor
from api_response import APIResponse
from tracking import TokenUsage
from token_usage_persistence import TokenUsagePersistence
from utils.file_utils import get_image_files

logger = logging.getLogger(__name__)


def process_receipt(image_path: str, processor: ReceiptProcessor) -> APIResponse:
    """Process a single receipt image"""
    return processor.process_directly(image_path)


def print_processing_result(result: APIResponse, image_files: List[Path], index: int):
    """Print the result of processing a single image"""
    logger.info(f"Processing image: {image_files[index]}")

    if result.status == 'success':
        logger.info("✅ SUCCESS")
        if result.data:
            logger.info(f"Parsed Receipt: {result.data}")
    else:
        logger.error("❌ FAILED")
        if result.error:
            logger.error(f"Error: {result.error}")


def print_batch_summary(successful: int, failed: int, total: int):
    """Print batch processing summary"""
    logger.info("=" * 50)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Success Rate: {(successful / total * 100):.1f}%")


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
        result = process_single_image(image_path, processor)

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
            logger.info("✅ SUCCESS")
        else:
            failed_processes += 1
            logger.error("❌ FAILED")

        # Print individual result
        print_processing_result(result, image_files, i-1)

    return successful_processes, failed_processes, total_token_usage


def process_single_image(image_path: Path, processor: ReceiptProcessor) -> APIResponse:
    """Process a single image file"""
    logger.info(f"Processing image: {image_path}")

    result = processor.process_directly(str(image_path))

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

        logger.info("✅ SUCCESS")

        # Print parsed receipt if available
        if 'parsed_receipt' in data_dict:
            logger.info(f"Parsed Receipt: {data_dict['parsed_receipt']}")
        elif hasattr(result.data, 'model_dump'):
            # If it's a ReceiptModel, show the model data
            logger.info(f"Parsed Receipt: {data_dict}")
    else:
        logger.error("❌ FAILED")
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
