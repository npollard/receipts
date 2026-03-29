# Receipts

Leverage AI to take images of receipts and tell me where all my grocery money is going.

## LangChain/LangGraph Setup with Modular Architecture

This project uses LangChain/LangGraph for workflow management with modular OOP architecture.

### Prerequisites

- Python 3.14+
- OpenAI API key
- Tesseract OCR

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
- **macOS**: `brew install tesseract`
- **Ubuntu**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from [Tesseract UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

3. Set up environment:
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

4. Place receipt images in `imgs/` directory (supports .jpg files)

### Running

```bash
python main.py
```

### Architecture

The project is split into modular components:

**Core Modules:**
- **`models.py`** - Pydantic data models (Receipt, ReceiptItem)
- **`api_response.py`** - Standardized API response structure
- **`token_tracking.py`** - Token usage tracking utilities
- **`image_processing.py`** - OCR and image preprocessing
- **`ai_parsing.py`** - OpenAI receipt parsing with validation
- **`workflow.py`** - LangGraph workflow orchestration
- **`receipt_processor.py`** - Main processor facade
- **`main.py`** - Entry point and CLI interface

**Key Classes:**
- **`ReceiptItem`** - Individual receipt items with validation
- **`Receipt`** - Complete receipt with total validation
- **`APIResponse`** - Standardized success/failure responses
- **`TokenUsage`** - Tracks OpenAI API usage and costs
- **`OCRProcessor`** - OpenCV/Tesseract image processing
- **`ReceiptParser`** - OpenAI GPT-4o-mini with Pydantic validation
- **`WorkflowOrchestrator`** - Coordinates processing workflow
- **`ReceiptProcessor`** - Main facade interface

**Processing Workflow:**
1. **OCR Extraction**: Extract text from receipt images using Tesseract
2. **AI Parsing**: Use OpenAI GPT-4o-mini to parse OCR text into structured JSON
3. **Validation**: Validate parsed data with Pydantic models
4. **Token Tracking**: Monitor OpenAI API usage and costs

### Token Usage

The application tracks OpenAI token usage and provides:
- Input/output token counts
- Request count
- Session duration
- Estimated cost (GPT-4o-mini pricing)
- Success/failure statistics

### Error Handling

Uses structured error handling with:
- **Pydantic validation** - Data integrity checks
- **APIResponse pattern** - Consistent success/failure format
- **Validation errors** - Detailed error information with raw data
- **Graceful degradation** - Continues processing even with individual failures

### Dependencies

- OpenCV for image preprocessing
- Tesseract for OCR
- LangChain/LangGraph for workflow orchestration
- OpenAI for receipt parsing
- Pydantic for data validation
- python-dotenv for environment management
