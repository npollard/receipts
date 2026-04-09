import warnings
import sys
import os
import time
from pathlib import Path
from unittest import mock

# Disable EasyOCR model downloads in test environment
# Must be set BEFORE any easyocr imports
os.environ['EASYOCR_MODULE_PATH'] = '/dev/null/easyocr_models'

# Mock EasyOCR Reader to prevent any model downloads or initialization
# This is a safety net in case something imports easyocr directly
_mock_easyocr_patcher = None

def _mock_easyocr_reader(*args, **kwargs):
    """Mock EasyOCR Reader that returns a no-op object"""
    class MockReader:
        def readtext(self, img, **kwargs):
            return []
        def detect(self, img, **kwargs):
            return [], []
        def recognize(self, img, **kwargs):
            return []
    return MockReader()

try:
    import easyocr
    _mock_easyocr_patcher = mock.patch('easyocr.Reader', _mock_easyocr_reader)
    _mock_easyocr_patcher.start()
except ImportError:
    pass  # easyocr not installed, nothing to mock

# Add src to Python path for imports (src layout)
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also add current directory for relative imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from urllib3.exceptions import NotOpenSSLWarning
except Exception:  # pragma: no cover
    NotOpenSSLWarning = Warning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1.1\+.*",
)

# Global OCR mocking - prevents any real OCR execution in tests
def _create_mock_ocr_response():
    """Create a mock OCR response with observability"""
    from services.ocr_service import OCRObservability

    start_time = time.time()
    end_time = start_time + 0.1  # Fake 100ms duration

    obs = OCRObservability(
        method='mock',
        start_time=start_time,
        end_time=end_time,
        quality_score=0.85,
        text_length=150,
        confidence_threshold=0.5,
    )
    return "FAKE OCR TEXT FOR TESTING - Sample receipt content", obs


# Monkeypatch OCRService before any tests run
# This prevents EasyOCR from ever being initialized
from services.ocr_service import OCRService
OCRService.extract_text = lambda self, image_path, use_vision_fallback=False: "FAKE OCR TEXT FOR TESTING - Sample receipt content"
OCRService.extract_text_with_observability = lambda self, image_path, use_vision_fallback=False: _create_mock_ocr_response()
