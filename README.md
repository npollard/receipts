# Receipts

AI-powered receipt processing with production-grade LLM workflows—reliability, validation, error-driven retries, and controlled concurrency.

## Features

- **Hybrid OCR**: Local EasyOCR + OpenAI Vision fallback
- **Error-driven retries**: 3 adaptive strategies based on failure severity
- **Clean architecture**: Domain layer isolated from infrastructure
- **Controlled concurrency**: Single-layer parallelism with runtime guards
- **Full observability**: Token tracking, cost estimation, per-image timing
- **Idempotent processing**: Content hash deduplication

## Quick Start

```bash
# Setup
git clone <repository-url>
cd receipts
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Configure
export OPENAI_API_KEY="your-key-here"
mkdir -p imgs  # Place receipt images here

# Run
python main.py                      # Dev mode (single-threaded)
EXECUTION_MODE=local python main.py  # 2 workers
```

## Architecture (High Level)

```
OCR → LLM Parsing → Validation → Retry → Persistence
```

- **OCR**: EasyOCR (local) → Vision API (fallback if quality < 0.25)
- **LLM**: GPT-4o-mini structures OCR text into JSON
- **Validation**: Pydantic + custom rules preserve partial results
- **Retry**: Self-correction → RAG → OCR fallback based on error severity
- **Concurrency**: Single-layer (BatchProcessingService) with enforced thread limits

## Example Output

```
==============================
RECEIPT: receipt_001.jpg
==============================

STATUS: ✅ SUCCESS
RETRIES: LLM_SELF_CORRECTION

TOTAL: $47.83

ITEMS (3):
- Milk .... $3.49
- Eggs .... $5.99
- Bread .... $2.50

TOKENS: 1670 | OCR: 245ms | Total: 1450ms
```

## Learn More

See `/docs` for:
- [Concurrency model](docs/concurrency.md)
- [Retry strategies](docs/retry-strategies.md)
- [Architecture deep dive](docs/architecture.md)
- [Testing guide](docs/testing.md)