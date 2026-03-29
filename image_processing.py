"""Image processing utilities"""

import logging
import base64
from abc import ABC, abstractmethod
from typing import Any
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

logger = logging.getLogger(__name__)


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


class VisionProcessor(ImageProcessor):
    """Concrete implementation of image processing using OpenAI Vision API"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.vision_llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self._preprocess_chain = RunnableLambda(self._preprocess_image)
        self._ocr_chain = RunnableLambda(self._extract_vision_text)
        logger.info(f"Initialized VisionProcessor with model: {model_name}")

    def preprocess(self, image_path: str) -> Any:
        """Public interface for preprocessing"""
        return self._preprocess_image(image_path)

    def extract_text(self, image_path: str) -> str:
        """Public interface for OCR extraction"""
        return self._extract_vision_text(image_path)

    def _preprocess_image(self, image_path: str) -> Any:
        """Internal preprocessing logic - encode image for vision API"""
        logger.debug(f"Encoding image for vision API: {image_path}")

        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                logger.debug(f"Successfully encoded image: {len(base64_image)} characters")
                return base64_image
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            raise

    def _extract_vision_text(self, image_path: str) -> str:
        """Internal OCR extraction logic using OpenAI Vision"""
        logger.debug(f"Extracting text from image using Vision API: {image_path}")

        try:
            # Encode image
            base64_image = self._preprocess_image(image_path)

            # Create vision prompt with structured output requirements
            vision_prompt = """
Extract all text from this receipt image exactly as it appears.
Focus on:
- Store name/location
- Date and time
- All individual items with prices
- Subtotals and final total
- Any other visible text (tax, phone numbers, etc.)

Return the extracted text in a clean, readable format.
Preserve the original layout and formatting as much as possible.
"""

            # Create message with image
            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": vision_prompt
                    }
                ]
            )

            # Call vision API
            response = self.vision_llm.invoke([message])
            extracted_text = response.content.strip()

            logger.info(f"Extracted {len(extracted_text)} characters from {image_path}")
            logger.debug(f"Extracted text: {extracted_text[:200]}...")

            return extracted_text

        except Exception as e:
            logger.error(f"Vision API extraction failed for {image_path}: {str(e)}")
            raise

    @tool
    def preprocess_image_tool(self, image_path: str) -> Any:
        """LangChain tool for image preprocessing"""
        return self.preprocess(image_path)

    @tool
    def extract_vision_text_tool(self, image_path: str) -> str:
        """LangChain tool for vision text extraction"""
        return self.extract_text(image_path)
