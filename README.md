# Receipts

Leverage AI to take images of receipts and tell me where all my grocery money is going.

## Docker Setup with LangChain/LangGraph

This project has been dockerized and uses LangChain/LangGraph for workflow management.

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### Setup

1. Copy the environment file:
```bash
cp .env.example .env
```

2. Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=your_actual_api_key_here
```

3. Place receipt images in the `imgs/` directory (supports .jpg files)

### Running

Option 1: Using Docker Compose (recommended):
```bash
docker-compose up --build
```

Option 2: Using Docker directly:
```bash
docker build -t receipts .
docker run -v $(pwd)/imgs:/app/imgs --env-file .env receipts
```

### Architecture

The application uses LangChain/LangGraph with the following workflow:
1. **OCR Extraction**: Extract text from receipt images using Tesseract
2. **AI Parsing**: Use OpenAI GPT-4o-mini to parse OCR text into structured JSON
3. **Results Output**: Print parsed receipt data

### Dependencies

- OpenCV for image preprocessing
- Tesseract for OCR
- LangChain/LangGraph for workflow orchestration
- OpenAI for receipt parsing
