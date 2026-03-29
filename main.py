"""Main entry point for receipt processing"""

import logging
from pathlib import Path
from receipt_processor import ReceiptProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
                "parse_result": processor.ai_parser.parse(x['ocr_text'])
            })
            | RunnableLambda(lambda x: {
                "image_path": x["image_path"],
                "ocr_text": x["ocr_text"],
                "result": x["parse_result"]
            })
        )

        result = processing_chain.invoke(image_path)

        logger.info(f"Chain processing: {result['image_path']}")
        logger.info(f"OCR Text: {result['ocr_text'][:100]}..." if len(result['ocr_text']) > 100 else f"OCR Text: {result['ocr_text']}")

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
        error_msg = f"Chain processing error: {str(e)}"
        logger.error(error_msg)
        return APIResponse.failure(error_msg)


def _is_valid_json(text: str) -> bool:
    """Helper function to validate JSON"""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


if __name__ == "__main__":
    filenames = list(Path('./imgs').rglob('*.jpg'))
    processor = ReceiptProcessor()

    logger.info("Starting receipt processing session...")
    logger.info(f"Found {len(filenames)} images to process")

    # Reset token usage for fresh session
    processor.reset_token_usage()

    successful_processes = 0
    failed_processes = 0

    for filename in filenames:
        logger.info(f"{'='*50}")

        # Choose processing method:
        # 1. LangGraph workflow (recommended)
        result = processor.process_with_langgraph(str(filename))

        # 2. Direct processing (simpler)
        # result = processor.process_directly(str(filename))

        # 3. Pure LangChain chains
        # result = process_image_with_langchain_chains(str(filename))

        # Track success/failure
        if isinstance(result, dict) and result.get('parsed_receipt', {}).get('status') == 'failed':
            failed_processes += 1
            logger.error(f"❌ FAILED: {result['parsed_receipt'].get('error', 'Unknown error')}")
        elif hasattr(result, 'status') and result.status == 'failed':
            failed_processes += 1
            logger.error(f"❌ FAILED: {result.error}")
        else:
            successful_processes += 1
            logger.info("✅ SUCCESS")

    # Print token usage summary at end
    logger.info("="*50)
    logger.info("SESSION COMPLETE")
    logger.info(f"✅ Successful: {successful_processes}")
    logger.info(f"❌ Failed: {failed_processes}")
    if filenames:
        logger.info(f"📊 Success Rate: {(successful_processes / len(filenames) * 100):.1f}%")
    else:
        logger.info("No files processed")

    summary = processor.get_token_usage_summary()
    logger.info(summary)
    logger.info("="*50)
