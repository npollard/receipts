"""Main entry point for receipt processing"""

import logging
import argparse
from pathlib import Path
from receipt_processor import ReceiptProcessor
from ai_parsing_with_persistence import ReceiptParser
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


def print_usage_summary():
    """Print token usage summary from persistent storage"""
    persistence = TokenUsagePersistence()
    summary = persistence.get_usage_summary()

    print("=" * 50)
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


def process_image_with_langgraph(image_path: str) -> Dict[str, Any]:
    """Process image using LangChain/LangGraph workflow with OOP architecture"""
    processor = ReceiptProcessor()
    return processor.process_with_langgraph(image_path)


def process_image_directly(image_path: str):
    """Process image directly using OOP architecture without LangGraph"""
    processor = ReceiptProcessor()
    return processor.process_directly(image_path)


def process_image_with_langchain_chains(image_path: str):
    """Alternative: Process image using pure LangChain chains with OOP components"""
    from langchain_core.runnables import RunnableLambda
    import json
    from api_response import APIResponse

    processor = ReceiptProcessor()

    try:
        # Create a processing chain using LangChain components and OOP objects
        processing_chain = (
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

        result = processing_chain.invoke(image_path)

        logger.info(f"Vision processing: {result['image_path']}")
        logger.info(f"Extracted Text: {result['ocr_text'][:100]}..." if len(result['ocr_text']) > 100 else f"Extracted Text: {result['ocr_text']}")

        if result["result"].status == "success":
            logger.info(f"Parsed Receipt: {json.dumps(result['result'].data, indent=2)}")
            return APIResponse.success({
                "image_path": result["image_path"],
                "ocr_text": result["ocr_text"],
                "parsed_receipt": result["result"].data
            })
        else:
            logger.error(f"Parsing Error: {result['result'].error}")
            return APIResponse.failure(f"Failed to parse receipt: {result['result'].error}")

    except Exception as e:
        error_msg = f"Vision chain processing error: {str(e)}"
        logger.error(error_msg)
        return APIResponse.failure(error_msg)


def _is_valid_json(text: str) -> bool:
    """Helper function to validate JSON"""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def print_results(result, successful_processes, failed_processes, image_files):
    """Print processing results"""
    if result.status == 'success':
        logger.info(f"Parsed Receipt: {json.dumps(result.data.get('parsed_receipt', {}), indent=2)}")
    else:
        logger.error(f"Parsing Error: {result.error}")

    # Print individual file status if processing multiple files
    if len(image_files) > 1:
        current_file = result.data.get('image_path', 'unknown') if result.data else 'unknown'
        for i, img_file in enumerate(image_files, 1):
            if str(img_file) == current_file:
                status = "✅ SUCCESS" if result.status == 'success' else "❌ FAILED"
                logger.info(f"  {i}. {img_file.name}: {status}")

    # Print final summary only once
    if len(image_files) > 1:
        logger.info("="*50)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"✅ Successful: {successful_processes}")
        logger.info(f"📊 Success Rate: {(successful_processes / len(image_files) * 100):.1f}%")


def main():
    """Main entry point"""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Process receipt images using AI')
    parser.add_argument('--usage-summary-only', action='store_true',
                       help='Show only persisted token usage summary without processing images')

    args = parser.parse_args()

    # Handle usage summary only request
    if args.usage_summary_only:
        print_usage_summary()
        return

    # Process all images in imgs directory
    imgs_dir = Path('imgs')
    if not imgs_dir.exists():
        logger.error("imgs directory not found")
        return

    # Get all image files
    image_files = list(imgs_dir.glob('*.jpg')) + list(imgs_dir.glob('*.jpeg')) + list(imgs_dir.glob('*.png'))

    if not image_files:
        logger.warning("No image files found in imgs directory")
        return

    logger.info(f"Found {len(image_files)} image files to process")

    # Process each image
    processor = ReceiptProcessor()
    successful_processes = 0
    failed_processes = 0
    total_token_usage = TokenUsage()  # Track token usage across all images

    for image_path in image_files:
        logger.info(f"Processing image: {image_path}")
        result = process_image_with_langchain_chains(str(image_path))

        # Track token usage from successful results
        if result.status == 'success' and result.data:
            token_usage_data = result.data.get('parsed_receipt', {}).get('_token_usage', {})
            if token_usage_data:
                total_token_usage.add_usage(
                    token_usage_data.get('input_tokens', 0),
                    token_usage_data.get('output_tokens', 0)
                )

        if result.status == 'success':
            successful_processes += 1
            logger.info("✅ SUCCESS")
        else:
            failed_processes += 1
            logger.error("❌ FAILED")
            print_results(result, successful_processes, failed_processes, image_files)

    # Print final summary
    logger.info("="*50)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"✅ Successful: {successful_processes}")
    logger.info(f"❌ Failed: {failed_processes}")
    logger.info(f"📊 Success Rate: {(successful_processes / len(image_files) * 100):.1f}%")

    # Print token usage summary
    logger.info("="*50)
    logger.info("TOKEN USAGE SUMMARY")
    logger.info(total_token_usage.get_summary())
    logger.info("="*50)

    # Save aggregated token usage to persistent storage
    if total_token_usage.get_total_tokens() > 0:
        persistence = TokenUsagePersistence()
        session_id = f"batch_session_{total_token_usage.get_total_tokens()}"
        persistence.save_usage(total_token_usage, session_id)
        logger.info(f"Saved batch token usage to persistent storage: {session_id}")

    # Show persisted usage summary after processing
    logger.info("="*50)
    logger.info("PERSISTED USAGE SUMMARY")
    print_usage_summary()
    logger.info("="*50)


if __name__ == "__main__":
    main()
