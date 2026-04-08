# Receipts

AI-powered receipt processing system exploring **production-grade LLM workflows**—reliability, validation, error-driven retries, and cost observability.

## Why This Project Exists

LLMs are powerful but unreliable in production. This codebase demonstrates patterns for making AI systems robust:

- **Deterministic + AI hybrid pipeline**: Local OCR (EasyOCR) first, LLM parsing second
- **Error-driven retry orchestration**: Three strategies triggered by validation failures
- **Graceful degradation**: Preserve partial results even when validation fails
- **Full observability**: Token usage tracking, cost estimation, structured logging

## Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Receipt   │───→│  EasyOCR     │───→│  GPT-4o-mini │───→│   Pydantic   │
│   Image     │    │  (local)     │    │   Parser     │    │ Validation   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                │
                        ┌───────────────────────────────────────┘
                        │ (on validation failure)
                        ▼
        ┌───────────────────────────────┐
        │    Error-Driven Retries       │
        │  1. LLM Self-Correction      │
        │  2. RAG with Focused Context │
        │  3. OCR Fallback (Vision API)│
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   SQLite / PostgreSQL Storage  │
        │   + Token Usage Persistence    │
        └───────────────────────────────┘
```

**Key Design Decisions:**
- **Local OCR first**: EasyOCR runs locally (free, fast), falls back to OpenAI Vision only when quality is low
- **Structured validation**: Pydantic models enforce schema + data types
- **Retry strategies adapt to error severity**: Small errors → self-correction; Large errors → RAG or OCR reprocessing
- **Always preserve parsed data**: Even failed validations return partial results for debugging

## Prerequisites

- Python 3.10+
- OpenAI API key (for LLM parsing and OCR fallback)

## Installation

```bash
# Clone and setup
git clone <repository-url>
cd receipts
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Configure
export OPENAI_API_KEY="your-key-here"
mkdir -p imgs # Place receipt images here
```

## Usage

```bash
# Process all images in imgs/
python main.py

# Show accumulated token usage and costs -- do not process images
python main.py --usage-summary-only

# Run without database (debug mode)
python main.py --no-db
```

Place receipt images (`.jpg`, `.jpeg`, `.png`) in `imgs/` and run.

## Pipeline Details

### 1. OCR: Local + Cloud Fallback

**EasyOCR** (default):
- Runs locally, no API cost
- GPU acceleration if available
- Quality scoring (0-1) determines if fallback needed

**OpenAI Vision** (fallback):
- Triggered when EasyOCR quality < 0.25
- Also used in retry strategy #3
- Higher cost but better on poor-quality images

### 2. LLM Parsing: GPT-4o-mini

Structures OCR text into JSON with fields:
- `date`: ISO format date
- `total`: Decimal amount
- `items`: List of `{description, quantity, price}`

### 3. Validation: Pydantic + Custom Rules

Validates:
- Required fields present (`date`, `total`, `items`)
- Data types correct (Decimal for money, list for items)
- At least 1 item in receipt
- Total is valid numeric

**On validation failure**: Parse result is preserved (not discarded) for retry analysis.

### 4. Retry Orchestration: Error-Driven

Up to **2 retries** per receipt, strategy selected by error severity:

| Strategy | Trigger | Description |
|----------|---------|-------------|
| **LLM Self-Correction** | Small/medium errors | Send original text + error back to LLM with correction prompt |
| **RAG + Focused Context** | Medium/large errors | Extract only lines near errors, re-parse with focused context |
| **OCR Fallback** | Low OCR quality or large errors | Re-extract with OpenAI Vision, then re-parse |

Token usage is **accumulated across all attempts**.

### 5. Persistence: Multi-Layer

**Database** (SQLite default, PostgreSQL supported):
- Users, receipts, items with full relationships
- Idempotency: Duplicate receipts detected by content hash
- Processing status tracking (pending → processing → completed/failed)

**Token Usage Persistence** (`token_usage.json`):
- Cross-session cost tracking
- Aggregated summaries (input/output/total/cost)

## Observability

### Token Usage & Costs

Pricing (GPT-4o-mini):
- Input: $0.15 per 1M tokens ($0.00015 per 1K)
- Output: $0.60 per 1M tokens ($0.00060 per 1K)

```bash
$ python main.py --usage-summary-only
==================================================
PERSISTED USAGE SUMMARY
==================================================
Total Sessions: 14
Total Input Tokens: 9850
Total Output Tokens: 5972
Total Tokens: 15822
Total Estimated Cost: $5.06
==================================================
```

### Output Format

```
==============================
RECEIPT: grocery_2024-01.jpg
==============================

STATUS: ✅ SUCCESS
RETRIES: LLM_SELF_CORRECTION

TOTAL: $47.83

ITEMS (3):
- Milk .... $3.49
- Eggs .... $5.99
- Bread .... $2.50

ITEM SUM: $11.98
MISMATCH: $35.85

TOKENS:
- Input: 1247
- Output: 423
- Total: 1670
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Integration tests (database, OCR, full pipeline)
python -m pytest tests/integration/ -v

# Specific modules
python -m pytest tests/test_api_response.py -v
python -m pytest tests/test_models.py -v
```

## Failure Modes Handled

| Issue | Handling |
|-------|----------|
| Invalid JSON from LLM | Retry with correction prompt |
| Missing required fields | Error classification → appropriate retry |
| 0 items extracted | Validation failure with partial data preserved |
| OCR too short/empty | Quality score triggers Vision fallback |
| Total ≠ sum(items) | Displayed as mismatch, not a hard failure |
| Duplicate receipt | Idempotency check prevents re-processing |
| Database unavailable | `--no-db` flag enables file-only mode |

## TODO

- [ ] Experiment with PaddleOCR for cost/quality comparison
- [ ] Mobile app for receipt upload
- [ ] REST API for external integrations
- [ ] Multi-user identity management (auth)
- [ ] Horizontal scaling architecture

## Project Structure

```
receipts/
├── main.py                    # CLI entry point
├── database_models.py         # SQLAlchemy models + DatabaseManager
├── receipt_persistence.py     # Receipt CRUD operations
├── token_tracking.py          # TokenUsage dataclass + cost calc
├── token_usage_persistence.py # JSON persistence for usage
├── pipeline/
│   └── processor.py           # ReceiptProcessor orchestration
├── domain/parsing/
│   └── receipt_parser.py      # LLM parsing + retry strategies
├── services/
│   ├── ocr_service.py         # EasyOCR + Vision fallback
│   ├── batch_service.py       # Multi-image processing
│   └── token_service.py       # Usage aggregation
├── utils/
│   └── output_formatter.py    # Human-readable output
└── tests/
    ├── integration/           # End-to-end tests
    └── services/              # Unit tests
```
