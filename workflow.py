"""Workflow orchestration utilities"""

import logging
from typing import Dict, Any
import json

from image_processing import ImageProcessor
from ai_parsing import ReceiptParser
from token_tracking import TokenUsage
from api_response import APIResponse
from models import DecimalEncoder

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Orchestrates the receipt processing workflow with token tracking"""

    def __init__(self, image_processor: ImageProcessor, ai_parser: ReceiptParser):
        self.image_processor = image_processor
        self.ai_parser = ai_parser
        self.token_usage = TokenUsage()
        logger.info("Initialized WorkflowOrchestrator")

    def process_image(self, image_path: str) -> APIResponse:
        """Process a single image through complete workflow"""
        logger.info(f"Processing image: {image_path}")

        try:
            # Step 1: Extract OCR text
            ocr_text = self.image_processor.extract_text(image_path)
            logger.info(f"Extracted OCR text: {ocr_text[:100]}..." if len(ocr_text) > 100 else f"Extracted OCR text: {ocr_text}")

            # Step 2: Parse with AI (with token tracking)
            parse_result = self.ai_parser.parse_with_usage_tracking(ocr_text, self.token_usage)

            # Step 3: Output results
            if parse_result.status == "success":
                logger.info(f"Successfully parsed receipt: {image_path}")
                logger.debug(f"Parsed data: {json.dumps(parse_result.data, cls=DecimalEncoder, indent=2)}")
                return APIResponse.success({
                    "image_path": image_path,
                    "ocr_text": ocr_text,
                    "parsed_receipt": parse_result.data
                })
            else:
                logger.error(f"Parsing failed for {image_path}: {parse_result.error}")
                # Include validation details in error response for debugging
                return APIResponse.failure(f"Failed to parse receipt: {parse_result.error}")

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(f"Error processing {image_path}: {error_msg}")
            return APIResponse.failure(error_msg)

    def get_token_usage_summary(self) -> str:
        """Get current token usage summary"""
        return self.token_usage.get_summary()

