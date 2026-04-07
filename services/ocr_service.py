"""OCR service for extracting text from images"""

import logging
from typing import Any
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

from utils.file_utils import encode_file_base64

logger = logging.getLogger(__name__)


class OCRService:
    """Service for OCR text extraction using OpenAI Vision API"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.vision_llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self._preprocess_chain = RunnableLambda(self._preprocess_image)
        self._ocr_chain = RunnableLambda(self._extract_vision_text)
        logger.info(f"Initialized OCRService with model: {model_name}")

    def preprocess_image(self, image_path: str) -> Any:
        """Preprocess image for OCR"""
        return self._preprocess_image(image_path)

    def extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        return self._extract_vision_text(image_path)

    def _preprocess_image(self, image_path: str) -> dict:
        """Convert image to base64 for vision API"""
        try:
            base64_image = encode_file_base64(image_path)

            return {
                "image_path": image_path,
                "base64_image": base64_image
            }
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise

    def _extract_vision_text(self, image_input: Any) -> str:
        """Extract text using OpenAI Vision API"""
        if isinstance(image_input, str):
            # If it's a string path, preprocess first
            processed = self._preprocess_image(image_input)
        else:
            processed = image_input

        vision_prompt = """
        Extract all text from this receipt image. Return ONLY the extracted text
        without any additional commentary, formatting, or explanations.
        Preserve the layout and structure as much as possible.
        """

        message = HumanMessage(
            content=[
                {"type": "text", "text": vision_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{processed['base64_image']}",
                        "detail": "auto"
                    }
                }
            ]
        )

        try:
            response = self.vision_llm.invoke([message])
            extracted_text = response.content.strip()

            logger.info(f"Extracted {len(extracted_text)} characters from {processed['image_path']}")
            return extracted_text

        except Exception as e:
            logger.error(f"Vision API error for {processed['image_path']}: {str(e)}")
            raise
