"""Unit tests for VisionProcessor image processing"""

import pytest
from unittest.mock import Mock, patch, mock_open
from image_processing import VisionProcessor
from base64 import b64encode


def test_vision_processor_initialization():
    """Test VisionProcessor initialization with default model"""
    with patch('image_processing.ChatOpenAI'):
        processor = VisionProcessor()
        assert processor.vision_llm is not None


def test_vision_processor_initialization_custom_model():
    """Test VisionProcessor initialization with custom model"""
    with patch('image_processing.ChatOpenAI') as mock_llm, \
         patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        VisionProcessor(model_name="gpt-4o")
        mock_llm.assert_called_once_with(
            model="gpt-4o",
            temperature=0.0,
            api_key="test-key"
        )


def test_preprocess_image_success():
    """Test successful image preprocessing"""
    fake_image_data = b"fake-image-bytes"
    expected_base64 = b64encode(fake_image_data).decode('utf-8')

    with patch('image_processing.ChatOpenAI'):
        processor = VisionProcessor()

        with patch("builtins.open", mock_open(read_data=fake_image_data)):
            result = processor.preprocess("test.jpg")

            assert result == expected_base64


def test_preprocess_image_file_not_found():
    """Test preprocessing with missing file"""
    with patch('image_processing.ChatOpenAI'):
        processor = VisionProcessor()

        with patch("builtins.open", side_effect=FileNotFoundError()):
            with pytest.raises(FileNotFoundError):
                processor.preprocess("missing.jpg")


def test_extract_vision_text_success():
    """Test successful vision text extraction"""
    fake_image_data = b"fake-image-bytes"
    expected_ocr = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"

    with patch('image_processing.ChatOpenAI') as mock_llm:
        mock_response = Mock()
        mock_response.content = expected_ocr
        mock_llm.return_value.invoke.return_value = mock_response

        processor = VisionProcessor()

        with patch("builtins.open", mock_open(read_data=fake_image_data)):
            result = processor.extract_text("test.jpg")

            assert result == expected_ocr


def test_extract_vision_text_api_error():
    """Test vision text extraction with API error"""
    fake_image_data = b"fake-image-bytes"

    with patch('image_processing.ChatOpenAI') as mock_llm:
        mock_llm.return_value.invoke.side_effect = Exception("API Error")

        processor = VisionProcessor()

        with patch("builtins.open", mock_open(read_data=fake_image_data)):
            with pytest.raises(Exception, match="API Error"):
                processor.extract_text("test.jpg")


def test_extract_vision_text_uses_correct_prompt():
    """Test that vision extraction uses the correct prompt"""
    fake_image_data = b"fake-image-bytes"
    expected_ocr = "Some text"

    with patch('image_processing.ChatOpenAI') as mock_llm:
        mock_response = Mock()
        mock_response.content = expected_ocr
        mock_llm.return_value.invoke.return_value = mock_response

        processor = VisionProcessor()

        with patch("builtins.open", mock_open(read_data=fake_image_data)):
            processor.extract_text("test.jpg")

            # Verify the invoke was called with correct message structure
            call_args = mock_llm.return_value.invoke.call_args[0][0]
            message = call_args[0]

            assert len(message.content) == 2
            assert message.content[0]["type"] == "image_url"
            assert message.content[1]["type"] == "text"
            assert "Extract all text from this receipt image" in message.content[1]["text"]


def test_public_interface_methods():
    """Test that public interface methods delegate to internal methods"""
    with patch('image_processing.ChatOpenAI'):
        processor = VisionProcessor()

        # Mock internal methods
        processor._preprocess_image = Mock(return_value="base64-data")
        processor._extract_vision_text = Mock(return_value="extracted text")

        # Test public methods
        assert processor.preprocess("test.jpg") == "base64-data"
        assert processor.extract_text("test.jpg") == "extracted text"

        # Verify internal methods were called
        processor._preprocess_image.assert_called_once_with("test.jpg")
        processor._extract_vision_text.assert_called_once_with("test.jpg")
