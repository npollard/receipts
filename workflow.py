"""Workflow orchestration utilities"""

from typing import Dict, Any
import json

from image_processing import ImageProcessor
from ai_parsing import AIParser
from token_tracking import TokenUsage
from api_response import APIResponse
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
    image_path: str
    ocr_text: str
    parsed_receipt: Dict[str, Any]
    token_usage: TokenUsage


class WorkflowOrchestrator:
    """Orchestrates the receipt processing workflow with token tracking"""

    def __init__(self, image_processor: ImageProcessor, ai_parser: AIParser):
        self.image_processor = image_processor
        self.ai_parser = ai_parser
        self.token_usage = TokenUsage()

    def process_image(self, image_path: str) -> APIResponse:
        """Process a single image through the complete workflow"""
        print(f"\n\n****************")
        print(f"Processing: {image_path}")

        try:
            # Step 1: Extract OCR text
            ocr_text = self.image_processor.extract_text(image_path)
            print(f"OCR Text:\n{ocr_text}")
            print('****************')

            # Step 2: Parse with AI (with token tracking)
            parse_result = self.ai_parser.parse_with_usage_tracking(ocr_text, self.token_usage)

            # Step 3: Output results
            if parse_result.status == "success":
                print(f"Parsed Receipt:\n{json.dumps(parse_result.data, indent=2)}")
                return APIResponse.success({
                    "image_path": image_path,
                    "ocr_text": ocr_text,
                    "parsed_receipt": parse_result.data
                })
            else:
                print(f"Parsing Error: {parse_result.error}")
                return APIResponse.failure(f"Failed to parse receipt: {parse_result.error}")

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            print(f"Error: {error_msg}")
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
        state['ocr_text'] = self.image_processor.extract_text(state['image_path'])
        return state

    def _parse_receipt_node(self, state: State) -> State:
        """LangGraph node for AI parsing with token tracking"""
        parse_result = self.ai_parser.parse_with_usage_tracking(
            state['ocr_text'],
            self.token_usage
        )

        if parse_result.status == "success":
            state['parsed_receipt'] = parse_result.data
        else:
            state['parsed_receipt'] = {
                "status": "failed",
                "error": parse_result.error,
                "data": None
            }

        return state

    def _print_results_node(self, state: State) -> State:
        """LangGraph node for results output"""
        print(f"Parsed Receipt:\n{json.dumps(state['parsed_receipt'], indent=2)}")
        return state
