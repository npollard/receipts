import os
import cv2
import json
import pytesseract
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List, Protocol, Optional
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ValidationError, field_validator

load_dotenv()

@dataclass
class APIResponse:
    """Standardized API response structure"""
    status: str  # "success" or "failed"
    data: Any = None
    error: str = ""

    @classmethod
    def success(cls, data: Any) -> 'APIResponse':
        """Create a successful response"""
        return cls(status="success", data=data)

    @classmethod
    def failure(cls, error: str) -> 'APIResponse':
        """Create a failed response"""
        return cls(status="failed", error=error, data=None)

class ReceiptItem(BaseModel):
    """Individual receipt item model"""
    name: str
    category: str
    price: Decimal

    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v < 0:
            raise ValueError('Price must be non-negative')
        return v

class Receipt(BaseModel):
    """Main receipt data model"""
    date: Optional[str] = None
    items: List[ReceiptItem] = []
    total: Optional[Decimal] = None

    @field_validator('total')
    @classmethod
    def validate_total(cls, v, values):
        if v is not None and 'items' in values:
            calculated_total = sum(item.price for item in values['items'])
            if abs(v - calculated_total) > Decimal('0.01'):  # Allow small rounding differences
                raise ValueError(f'Total {v} does not match sum of items {calculated_total}')
        return v

@dataclass
class TokenUsage:
    """Track token usage across processing sessions"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    session_start: datetime = field(default_factory=datetime.now)

    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage from a single request"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.requests += 1

    def get_summary(self) -> str:
        """Get formatted summary of token usage"""
        duration = datetime.now() - self.session_start
        return f"""
Token Usage Summary:
- Input Tokens: {self.input_tokens:,}
- Output Tokens: {self.output_tokens:,}
- Total Tokens: {self.total_tokens:,}
- Requests: {self.requests}
- Session Duration: {duration}
- Est. Cost (GPT-4o-mini): ${(self.input_tokens * 0.00015 + self.output_tokens * 0.0006):.4f}
"""

class State(TypedDict):
    messages: Annotated[list, add_messages]
    image_path: str
    ocr_text: str
    parsed_receipt: Dict[str, Any]
    token_usage: TokenUsage

class ImageProcessor(ABC):
    """Abstract base class for image processing"""

    @abstractmethod
    def preprocess(self, image_path: str) -> Any:
        """Preprocess image for OCR"""
        pass

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from image"""
        pass

class OCRProcessor(ImageProcessor):
    """Concrete implementation of image processing using OpenCV and Tesseract"""

    def __init__(self, threshold: int = 150):
        self.threshold = threshold
        self._preprocess_chain = RunnableLambda(self._preprocess_image)
        self._ocr_chain = RunnableLambda(self._extract_ocr_text)

    def preprocess(self, image_path: str) -> Any:
        """Public interface for preprocessing"""
        return self._preprocess_image(image_path)

    def extract_text(self, image_path: str) -> str:
        """Public interface for OCR extraction"""
        return self._extract_ocr_text(image_path)

    def _preprocess_image(self, image_path: str) -> Any:
        """Internal preprocessing logic"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)[1]
        return img

    def _extract_ocr_text(self, image_path: str) -> str:
        """Internal OCR extraction logic"""
        processed_img = self._preprocess_image(image_path)
        return pytesseract.image_to_string(processed_img)

    @tool
    def preprocess_image_tool(self, image_path: str) -> Any:
        """LangChain tool for image preprocessing"""
        return self.preprocess(image_path)

    @tool
    def extract_ocr_text_tool(self, image_path: str) -> str:
        """LangChain tool for OCR extraction"""
        return self.extract_text(image_path)

class AIParser(ABC):
    """Abstract base class for AI parsing"""

    @abstractmethod
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse OCR text into structured data"""
        pass

class ReceiptParser(AIParser):
    """Concrete implementation of receipt parsing using OpenAI"""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self._system_prompt = "You are a receipt parser that returns valid JSON only."

    def parse(self, text: str) -> APIResponse:
        """Parse OCR text into structured receipt data"""
        prompt = self._build_prompt(text)
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            response = self.llm.invoke(messages)

            # Extract token usage from response
            usage_data = getattr(response, 'usage_metadata', {})
            input_tokens = usage_data.get('input_tokens', 0)
            output_tokens = usage_data.get('output_tokens', 0)

            parsed_data = json.loads(response.content)

            # Validate with Pydantic
            try:
                validated = Receipt(**parsed_data)
                result = validated.dict()

                # Add token usage to response data
                result["_token_usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }

                return APIResponse.success(result)

            except ValidationError as e:
                validation_error = {
                    "error": "validation_failed",
                    "details": e.errors(),
                    "raw": parsed_data
                }

                # Add token usage to validation error
                validation_error["_token_usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }

                return APIResponse.success(validation_error)

        except json.JSONDecodeError as e:
            return APIResponse.failure(f"Failed to parse JSON: {str(e)}")
        except Exception as e:
            return APIResponse.failure(f"Parsing error: {str(e)}")

    def parse_with_usage_tracking(self, text: str, token_usage: TokenUsage) -> APIResponse:
        """Parse with token usage tracking"""
        result = self.parse(text)

        # Update token usage tracker if successful
        if result.status == "success" and "_token_usage" in result.data:
            token_usage.add_usage(
                result.data["_token_usage"]["input_tokens"],
                result.data["_token_usage"]["output_tokens"]
            )

        return result

    def _build_prompt(self, ocr_text: str) -> str:
        """Build the parsing prompt"""
        return f"""
You are a receipt parser.

Given the OCR text below, extract:
- Date
- Items (name, category, price paid)
- Total

Return valid JSON only.

OCR TEXT:
{ocr_text}
"""

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

class ReceiptProcessor:
    """Main processor class that coordinates all components"""

    def __init__(self):
        self.image_processor = OCRProcessor()
        self.ai_parser = ReceiptParser()
        self.orchestrator = WorkflowOrchestrator(self.image_processor, self.ai_parser)

    def process_with_langgraph(self, image_path: str) -> Dict[str, Any]:
        """Process using LangGraph workflow"""
        app = self.orchestrator.create_langgraph_workflow()

        initial_state = {
            "messages": [],
            "image_path": image_path,
            "ocr_text": "",
            "parsed_receipt": {},
            "token_usage": self.orchestrator.token_usage
        }

        return app.invoke(initial_state)

    def process_directly(self, image_path: str) -> APIResponse:
        """Process directly without LangGraph"""
        return self.orchestrator.process_image(image_path)

    def get_token_usage_summary(self) -> str:
        """Get current token usage summary"""
        return self.orchestrator.get_token_usage_summary()

    def reset_token_usage(self):
        """Reset token usage tracking"""
        self.orchestrator.token_usage = TokenUsage()

def process_image_with_langgraph(image_path: str) -> Dict[str, Any]:
    """Process image using LangChain/LangGraph workflow with OOP architecture"""
    processor = ReceiptProcessor()
    return processor.process_with_langgraph(image_path)

def process_image_directly(image_path: str) -> APIResponse:
    """Process image directly using OOP architecture without LangGraph"""
    processor = ReceiptProcessor()
    return processor.process_directly(image_path)

def process_image_with_langchain_chains(image_path: str) -> APIResponse:
    """Alternative: Process image using pure LangChain chains with OOP components"""
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

        print(f"\n\n****************")
        print(f"Processing: {result['image_path']}")
        print(f"OCR Text:\n{result['ocr_text']}")
        print('****************')

        if result["result"].status == "success":
            print(f"Parsed Receipt:\n{json.dumps(result['result'].data, indent=2)}")
            return APIResponse.success({
                "image_path": result["image_path"],
                "ocr_text": result["ocr_text"],
                "parsed_receipt": result["result"].data
            })
        else:
            print(f"Parsing Error: {result['result'].error}")
            return APIResponse.failure(f"Failed to parse receipt: {result['result'].error}")

    except Exception as e:
        error_msg = f"Chain processing error: {str(e)}"
        print(f"Error: {error_msg}")
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

    print("Starting receipt processing session...")
    print(f"Found {len(filenames)} images to process")

    # Reset token usage for fresh session
    processor.reset_token_usage()

    successful_processes = 0
    failed_processes = 0

    for filename in filenames:
        print(f"\n{'='*50}")

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
            print(f"❌ FAILED: {result['parsed_receipt'].get('error', 'Unknown error')}")
        elif hasattr(result, 'status') and result.status == 'failed':
            failed_processes += 1
            print(f"❌ FAILED: {result.error}")
        else:
            successful_processes += 1
            print("✅ SUCCESS")

    # Print token usage summary at the end
    print("\n" + "="*50)
    print("SESSION COMPLETE")
    print(f"✅ Successful: {successful_processes}")
    print(f"❌ Failed: {failed_processes}")
    print(f"📊 Success Rate: {(successful_processes / len(filenames) * 100):.1f}%" if filenames else "No files processed")
    print(processor.get_token_usage_summary())
    print("="*50)
