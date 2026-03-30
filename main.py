"""Main entry point for receipt processing"""

import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from receipt_processor import ReceiptProcessor
from ai_parsing import ReceiptParser
from token_usage_persistence import TokenUsagePersistence
from token_tracking import TokenUsage
from langchain_core.runnables import RunnableLambda
import json
from api_response import APIResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_usage_summary(show_persisted: bool = False):
    """Print token usage summary from persistent storage"""
    persistence = TokenUsagePersistence()
    summary = persistence.get_usage_summary()

    print("=" * 50)
    if show_persisted:
        print("PERSISTED USAGE SUMMARY")
    else:
        print("TOKEN USAGE SUMMARY")
    print("=" * 50)
    print(f"Total Sessions: {summary['total_sessions']}")
    print(f"Total Input Tokens: {summary['total_input_tokens']:,}")
    print(f"Total Output Tokens: {summary['total_output_tokens']:,}")
    print(f"Total All Tokens: {summary['total_tokens']:,}")
    print(f"Total Estimated Cost: ${summary['total_estimated_cost']:.4f}")
    print("=" * 50)

    if 'recent_sessions' in summary:
        print("RECENT SESSIONS:")
        for i, session in enumerate(summary['recent_sessions'], 1):
            print(f"  {i}. Session {session.get('session_id', 'unknown')}: "
                  f"{session['total_tokens']} tokens, "
                  f"${session.get('estimated_cost', 0):.4f}")


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from directory"""
    return list(directory.glob('*.jpg')) + list(directory.glob('*.jpeg')) + list(directory.glob('*.png'))


def create_processing_chain(processor: ReceiptProcessor) -> RunnableLambda:
    """Create LangChain processing pipeline"""
    return (
        RunnableLambda(lambda x: {"image_path": x})
        | RunnableLambda(lambda x: {**x, "ocr_text": processor.image_processor.extract_text(x["image_path"])})
        | RunnableLambda(lambda x: {
            **x,
            "parse_result": processor.ai_parser.parse_text(x['ocr_text'])
        })
        | RunnableLambda(lambda x: {
            "image_path": x["image_path"],
            "ocr_text": x["ocr_text"],
            "result": x["parse_result"]
        })
    )


def process_single_image(image_path: Path, processor: ReceiptProcessor) -> APIResponse:
    """Process a single image using LangChain chains"""
    try:
        processing_chain = create_processing_chain(processor)
        result = processing_chain.invoke(str(image_path))

        logger.info(f"Processing image: {image_path}")
        logger.info(f"Extracted Text: {result['ocr_text'][:100]}..."
                   if len(result['ocr_text']) > 100 else f"Extracted Text: {result['ocr_text']}")

        if result["result"].status == "success":
            return APIResponse.success({
                "image_path": str(image_path),
                "ocr_text": result["ocr_text"],
                "parsed_receipt": result["result"].data
            })
        else:
            return APIResponse.failure(f"Failed to parse receipt: {result['result'].error}")

    except Exception as e:
        error_msg = f"Vision chain processing error: {str(e)}"
        logger.error(error_msg)
        return APIResponse.failure(error_msg)


def print_processing_result(result: APIResponse, image_files: List[Path], index: int = None):
    """Print processing result for a single image"""
    if result.status == 'success':
        logger.info(f"Parsed Receipt: {json.dumps(result.data.get('parsed_receipt', {}), indent=2)}")
    else:
        logger.error(f"Parsing Error: {result.error}")

    # Print individual file status if processing multiple files
    if len(image_files) > 1 and index is not None:
        status = "✅ SUCCESS" if result.status == 'success' else "❌ FAILED"
        logger.info(f"  {index}. {image_files[index].name}: {status}")


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


def process_batch_images(image_files: List[Path]) -> tuple[int, int, TokenUsage]:
    """Process multiple images and return success/failure counts and token usage"""
    processor = ReceiptProcessor()
    successful_processes = 0
    failed_processes = 0
    total_token_usage = TokenUsage()

    for i, image_path in enumerate(image_files, 1):
        result = process_single_image(image_path, processor)

        # Track token usage from successful results
        if result.status == 'success' and result.data:
            token_usage_data = result.data.get('parsed_receipt', {}).get('_token_usage', {})
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


def validate_and_get_image_files(directory: Path) -> List[Path]:
    """Validate directory exists and return image files"""
    if not directory.exists():
        logger.error("imgs directory not found")
        return []

    image_files = get_image_files(directory)
    if not image_files:
        logger.warning("No image files found in imgs directory")
        return []

    logger.info(f"Found {len(image_files)} image files to process")
    return image_files


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Process receipt images using AI')
    parser.add_argument('--usage-summary-only', action='store_true',
                       help='Show only persisted token usage summary without processing images')

    args = parser.parse_args()

    # Handle usage summary only request
    if args.usage_summary_only:
        print_usage_summary(show_persisted=False)
        return

    # Process all images in imgs directory
    imgs_dir = Path('imgs')
    image_files = validate_and_get_image_files(imgs_dir)

    if not image_files:
        return

    # Process batch
    successful, failed, token_usage = process_batch_images(image_files)

    # Print batch summary
    print_batch_summary(successful, failed, len(image_files))

    # Print current batch token usage
    print_token_usage_summary(token_usage)

    # Save token usage to persistence
    save_token_usage_to_persistence(token_usage)

    # Show persisted usage summary after processing
    print_usage_summary(show_persisted=True)


if __name__ == "__main__":
    main()
