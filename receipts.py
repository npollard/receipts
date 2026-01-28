import os
import cv2
import json
import pytesseract
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.runnables import RunnableLambda

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    image_path: str
    ocr_text: str
    parsed_receipt: Dict[str, Any]

class ReceiptProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Create LangChain runnables for image processing
        self.preprocess_chain = RunnableLambda(self._preprocess_image)
        self.ocr_chain = RunnableLambda(self._extract_ocr_text)
        
    @tool
    def preprocess_image(self, image_path: str) -> any:
        """Preprocess image for OCR using OpenCV"""
        return self._preprocess_image(image_path)
    
    def _preprocess_image(self, image_path: str) -> any:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]
        return img
    
    @tool
    def extract_ocr_text(self, image_path: str) -> str:
        """Extract text from image using Tesseract OCR"""
        processed_img = self._preprocess_image(image_path)
        return pytesseract.image_to_string(processed_img)
    
    def _extract_ocr_text(self, image_path: str) -> str:
        processed_img = self._preprocess_image(image_path)
        return pytesseract.image_to_string(processed_img)
    
    def extract_ocr_text_node(self, state: State) -> State:
        """LangChain node for OCR extraction using RunnableLambda"""
        print(f"\n\n****************")
        print(f"Processing: {state['image_path']}")
        
        # Use LangChain chain for OCR processing
        ocr_chain = self.ocr_chain
        raw_text = ocr_chain.invoke(state['image_path'])
        
        print(f"OCR Text:\n{raw_text}")
        print('****************')
        
        state['ocr_text'] = raw_text
        return state
    
    def parse_receipt_with_ai(self, state: State) -> State:
        """LangChain node for AI parsing using ChatOpenAI"""
        prompt = f"""
You are a receipt parser.

Given the OCR text below, extract:
- Date
- Items (name, category, price paid)
- Total

Return valid JSON only.

OCR TEXT:
{state['ocr_text']}
"""
        
        messages = [
            SystemMessage(content="You are a receipt parser that returns valid JSON only."),
            HumanMessage(content=prompt)
        ]
        
        # Use LangChain LLM for parsing
        response = self.llm.invoke(messages)
        
        try:
            parsed_data = json.loads(response.content)
            state['parsed_receipt'] = parsed_data
        except json.JSONDecodeError:
            state['parsed_receipt'] = {"error": "Failed to parse JSON", "raw_response": response.content}
        
        return state
    
    def print_results(self, state: State) -> State:
        """LangChain node for results output"""
        print(f"Parsed Receipt:\n{json.dumps(state['parsed_receipt'], indent=2)}")
        return state

def create_receipt_graph():
    processor = ReceiptProcessor()
    
    workflow = StateGraph(State)
    
    # Add nodes using LangChain components
    workflow.add_node("extract_ocr", processor.extract_ocr_text_node)
    workflow.add_node("parse_receipt", processor.parse_receipt_with_ai)
    workflow.add_node("print_results", processor.print_results)
    
    # Define the workflow edges
    workflow.add_edge(START, "extract_ocr")
    workflow.add_edge("extract_ocr", "parse_receipt")
    workflow.add_edge("parse_receipt", "print_results")
    workflow.add_edge("print_results", END)
    
    return workflow.compile()

def process_image_with_langgraph(image_path: str):
    """Process image using LangChain/LangGraph workflow"""
    app = create_receipt_graph()
    
    initial_state = {
        "messages": [],
        "image_path": image_path,
        "ocr_text": "",
        "parsed_receipt": {}
    }
    
    result = app.invoke(initial_state)
    return result

def process_image_with_langchain_chains(image_path: str):
    """Alternative: Process image using pure LangChain chains"""
    processor = ReceiptProcessor()
    
    # Create a processing chain using LangChain components
    processing_chain = (
        RunnableLambda(lambda x: {"image_path": x})
        | RunnableLambda(lambda x: {**x, "ocr_text": processor.ocr_chain.invoke(x["image_path"])})
        | RunnableLambda(lambda x: {
            **x, 
            "parsed_receipt": processor.llm.invoke([
                SystemMessage(content="You are a receipt parser that returns valid JSON only."),
                HumanMessage(content=f"""
You are a receipt parser.

Given the OCR text below, extract:
- Date
- Items (name, category, price paid)
- Total

Return valid JSON only.

OCR TEXT:
{x['ocr_text']}
""")
            ])
        })
        | RunnableLambda(lambda x: {
            "image_path": x["image_path"],
            "ocr_text": x["ocr_text"],
            "parsed_receipt": json.loads(x["parsed_receipt"].content) if _is_valid_json(x["parsed_receipt"].content) else {"error": "Failed to parse JSON", "raw_response": x["parsed_receipt"].content}
        })
    )
    
    result = processing_chain.invoke(image_path)
    
    print(f"\n\n****************")
    print(f"Processing: {result['image_path']}")
    print(f"OCR Text:\n{result['ocr_text']}")
    print('****************')
    print(f"Parsed Receipt:\n{json.dumps(result['parsed_receipt'], indent=2)}")
    
    return result

def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


if __name__ == "__main__":
    filenames = Path('./imgs').rglob('*.jpg')
    for filename in filenames:
        process_image_with_langgraph(str(filename))
