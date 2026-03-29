"""Main entry point for receipt processing"""

import logging
from pathlib import Path
from receipt_processor import ReceiptProcessor
from ai_parsing_with_persistence import ReceiptParser
from token_usage_persistence import TokenUsagePersistence

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
                "parse_result": processor.ai_parser.parse(x['ocr_text'])
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


def print_results(result, mode):
    if mode == 'direct':
        if result.status == 'success':
            logger.info(f"Parsed Receipt: {json.dumps(result.data, indent=2)}")
        else:
            logger.error(f"Parsing Error: {result.error}")
    elif mode == 'langgraph':
        if result.get('parsed_receipt', {}).get('status') == 'success':
            logger.info(f"Parsed Receipt: {json.dumps(result['parsed_receipt'].get('data', {}), indent=2)}")
        else:
            logger.error(f"Parsing Error: {result['parsed_receipt'].get('error', 'Unknown error')}")
    elif mode == 'chain':
        if result.status == 'success':
            logger.info(f"Parsed Receipt: {json.dumps(result.data.get('parsed_receipt', {}), indent=2)}")
        else:
            logger.error(f"Parsing Error: {result.error}")
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
