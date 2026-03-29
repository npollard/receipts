"""Workflow orchestration utilities"""

import logging
from typing import Dict, Any
import json

from image_processing import ImageProcessor
from ai_parsing_with_persistence import ReceiptParser
from token_tracking import TokenUsage
from api_response import APIResponse
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from models import DecimalEncoder

logger = logging.getLogger(__name__)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    image_path: str
    ocr_text: str
    parsed_receipt: Dict[str, Any]
    token_usage: TokenUsage


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

    def create_langgraph_workflow(self) -> StateGraph:
        """Create a LangGraph workflow using the OOP components"""
        workflow = StateGraph(State)

        workflow.add_node("extract_ocr", self._extract_ocr_node)
        workflow.add_node("parse_receipt", self._parse_receipt_node)
        workflow.add_node("print_results", self._print_results_node)

        workflow.add_edge(START, "extract_ocr")
        workflow.add_edge("extract_ocr", "parse_receipt")
        workflow.add_edge("parse_receipt", "print_results")
        workflow.add_edge("print_results", END)

        return workflow.compile()

    def _extract_ocr_node(self, state: State) -> State:
        """LangGraph node for OCR extraction"""
        logger.debug(f"OCR node processing: {state['image_path']}")
        state['ocr_text'] = self.image_processor.extract_text(state['image_path'])
        return state

    def _parse_receipt_node(self, state: State) -> State:
        """LangGraph node for AI parsing with token tracking"""
        logger.debug("Parse node processing OCR text")
        parse_result = self.ai_parser.parse_with_usage_tracking(
            state['ocr_text'],
            self.token_usage
        )

        if parse_result.status == "success":
            state['parsed_receipt'] = parse_result.data
            logger.debug("Parse node: success")
        else:
            # Store failure information in parsed_receipt
            state['parsed_receipt'] = {
                "status": "failed",
                "error": parse_result.error,
                "data": None
            }
            logger.debug("Parse node: failed")

        return state

    def _print_results_node(self, state: State) -> State:
        """LangGraph node for results output"""
        logger.info(f"Final result: {json.dumps(state['parsed_receipt'], cls=DecimalEncoder, indent=2)}")
        return state
