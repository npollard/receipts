# Receipts

Leverage AI to take images of receipts and tell me where all my grocery money is going.

## LangChain/LangGraph Setup

This project uses LangChain/LangGraph for workflow management with OOP architecture.

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

4. Place receipt images in the `imgs/` directory (supports .jpg files)

### Running

```bash
python receipts.py
```

### Architecture

The application uses LangChain/LangGraph with OOP design:

**Classes:**
- **`ImageProcessor`** (ABC) - Abstract interface for image processing
- **`OCRProcessor`** - OpenCV/Tesseract implementation
- **`AIParser`** (ABC) - Abstract interface for AI parsing
- **`ReceiptParser`** - OpenAI GPT-4o-mini implementation
- **`WorkflowOrchestrator`** - Coordinates workflow with token tracking
- **`ReceiptProcessor`** - Main facade class

**Processing Workflow:**
1. **OCR Extraction**: Extract text from receipt images using Tesseract
2. **AI Parsing**: Use OpenAI GPT-4o-mini to parse OCR text into structured JSON
3. **Token Tracking**: Monitor OpenAI API usage and costs

### Token Usage

The application tracks OpenAI token usage and provides:
- Input/output token counts
- Request count
- Session duration
- Estimated cost (GPT-4o-mini pricing)

### Dependencies

- OpenCV for image preprocessing
- Tesseract for OCR
- LangChain/LangGraph for workflow orchestration
- OpenAI for receipt parsing
- python-dotenv for environment management
