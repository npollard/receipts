# Receipts

AI-powered receipt processing system exploring **production-grade LLM workflows**—reliability, validation, error-driven retries, cost observability, and **controlled concurrency**.

## Why This Project Exists

LLMs are powerful but unreliable in production. This codebase demonstrates patterns for making AI systems robust:

- **Deterministic + AI hybrid pipeline**: Local OCR (EasyOCR) first, LLM parsing second
- **Error-driven retry orchestration**: Three strategies triggered by validation failures
- **Graceful degradation**: Preserve partial results even when validation fails
- **Full observability**: Token usage tracking, cost estimation, structured logging, per-image timing
- **Controlled concurrency**: Single-layer parallelism with explicit thread limits (no hidden CPU saturation)

## Concurrency Control

This application implements **strict, centralized concurrency control** to prevent CPU saturation and ensure predictable performance on resource-constrained environments like laptops.

### Single-Layer Parallelism Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Application Entrypoint (main.py)                            │
│  ├── Enforces thread limits BEFORE any imports               │
│  ├── OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS │
│  └── torch.set_num_threads()                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  BatchProcessingService (SINGLE parallelism layer)           │
│  ├── ProcessPoolExecutor(max_workers=MAX_WORKERS)            │
│  ├── Each worker: fresh OCR + Parser instances              │
│  └── No shared stateful objects across processes            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Worker Process                                             │
│  ├── OCRService (OCR_THREADS=1 when parallel)              │
│  ├── ReceiptParser                                          │
│  └── Sequential processing within worker                    │
└─────────────────────────────────────────────────────────────┘
```

### Execution Modes

| Mode | MAX_WORKERS | OCR_THREADS | Use Case |
|------|-------------|-------------|----------|
| **dev** (default) | 1 | 1 | Safe development, debugging, testing |
| **local** | 2 | 1 | Local batch processing on laptops |
| **cloud** | 4 | 1 | Cloud deployment with more resources |

### Environment Variables

```bash
# Execution mode (dev/local/cloud)
export EXECUTION_MODE=local

# Override defaults (optional)
export MAX_WORKERS=2        # Number of parallel workers
export OCR_THREADS=1        # Must be 1 when MAX_WORKERS > 1
```

### Runtime Guard

The application enforces **no nested parallelism**:
- If `MAX_WORKERS > 1` AND `OCR_THREADS > 1` → RuntimeError
- This ensures exactly ONE layer controls parallelism

### Performance Characteristics

- **Default mode (dev)**: ~100% CPU (single core), deterministic, safe for laptops
- **Local mode**: ~150-200% CPU (2 workers), good throughput without saturation
- **Cloud mode**: Scales based on available cores, ~400% CPU with 4 workers

## Architecture Overview

### Pipeline Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Receipt   │───→│  EasyOCR     │───→│  GPT-4o-mini │───→│   Pydantic   │
│   Image     │    │  (local)     │    │   Parser     │    │ Validation   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                        │                                            │
                        │ (low quality)                              │ (fails)
                        ▼                                            │
              ┌─────────────────┐                                    │
              │ OpenAI Vision   │◄───────────────────────────────────┘
              │ (fallback)      │          (retry with corrections)
              └─────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   SQLite / PostgreSQL Storage  │
        │   + Token Usage Persistence    │
        └───────────────────────────────┘
```

**Key Design Decisions:**
- **Interface-driven design**: `ImageProcessingInterface` and `ReceiptParsingInterface` enable clean dependency injection and testability
- **Local OCR first**: EasyOCR runs locally (free, fast), falls back to OpenAI Vision only when quality is low
- **Lazy initialization**: OCRService is only instantiated when needed, preventing heavy imports during tests
- **Structured validation**: Pydantic models enforce schema + data types
- **Retry strategies adapt to error severity**: Small errors → self-correction; Large errors → RAG or OCR reprocessing
- **Always preserve parsed data**: Even failed validations return partial results for debugging
- **Deterministic OCR scoring**: Structured scoring with clear thresholds (bad ≤0.4, medium ≥0.3, good ≥0.5)
- **Single-layer parallelism**: BatchProcessingService is the ONLY layer controlling concurrency
- **Thread limits enforced at entrypoint**: OMP, MKL, OpenBLAS, and torch threads are configured before any heavy imports

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

# Install package in editable mode (enables imports without PYTHONPATH)
pip install -e .

# Configure
export OPENAI_API_KEY="your-key-here"
mkdir -p imgs # Place receipt images here
```

## Usage

### Basic Usage

```bash
# Process all images in imgs/ (default: dev mode, single-threaded)
python main.py

# Run in local mode with limited parallelism (2 workers)
EXECUTION_MODE=local python main.py

# Run in cloud mode with more workers
EXECUTION_MODE=cloud python main.py

# Show accumulated token usage and costs -- do not process images
python main.py --usage-summary-only

# Run without database (debug mode)
python main.py --no-db
```

Place receipt images (`.jpg`, `.jpeg`, `.png`) in `imgs/` and run.

### Execution Mode Examples

```bash
# Development mode (default): Safe, single-threaded, deterministic
python main.py

# Local batch mode: 2 workers, good for laptop batch processing
EXECUTION_MODE=local python main.py

# Cloud mode: 4 workers, for deployment with more resources
EXECUTION_MODE=cloud python main.py

# Custom parallelism (dev mode with explicit overrides)
EXECUTION_MODE=dev MAX_WORKERS=2 OCR_THREADS=1 python main.py
```

### Output with Observability

The system now logs per-image processing details:

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

TOKENS:
- Input: 1247
- Output: 423
- Total: 1670

OCR: easyocr (245ms)
Processing: 1450ms
```

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

#### OCR Comparison

| Aspect | EasyOCR | OpenAI Vision |
|--------|---------|---------------|
| **Cost** | Free (local) | ~$0.001-0.005 per image |
| **Speed** | ~200-500ms | ~1000-3000ms |
| **Quality** | Good on clear images | Better on poor-quality images |
| **Fallback** | Primary method | Triggered when quality < 0.25 |
| **Thread Safety** | Respects thread limits | N/A (API call) |

The system automatically selects the best OCR method based on quality scoring, with per-image observability logged for analysis.

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

### Completed
- [x] **Controlled concurrency**: Centralized RuntimeConfig with single-layer parallelism
- [x] **Thread limits**: Enforced at entrypoint before any heavy imports
- [x] **Execution modes**: dev/local/cloud with different parallelism profiles
- [x] **OCR observability**: Per-image timing, OCR method tracking, quality scoring
- [x] **Runtime guard**: No nested parallelism enforcement

### In Progress / Planned
- [ ] Experiment with PaddleOCR for cost/quality comparison
- [ ] Mobile app for receipt upload
- [ ] REST API for external integrations
- [ ] Multi-user identity management (auth)
- [ ] Horizontal scaling architecture (cloud mode foundation laid)

## Project Structure

**Src Layout**: Application code lives under `src/` for clean imports and no module shadowing.

```
receipts/
├── main.py                    # CLI entry point
├── pyproject.toml             # Project configuration (pytest, dependencies)
├── src/                       # Application source code
│   ├── config/
│   │   ├── settings.py        # AppConfig, DatabaseConfig
│   │   └── runtime_config.py  # RuntimeConfig, ExecutionMode, thread limits
│   ├── pipeline/
│   │   └── processor.py       # ReceiptProcessor orchestration with DI
│   ├── domain/
│   │   ├── parsing/
│   │   │   └── receipt_parser.py  # LLM parsing + retry strategies
│   │   └── validation/
│   │       └── validation_service.py  # Pydantic validation
│   ├── services/
│   │   ├── ocr_service.py     # EasyOCR + Vision fallback (lazy init, thread-safe)
│   │   ├── batch_service.py   # Multi-image processing with ProcessPoolExecutor
│   │   └── token_service.py   # Usage aggregation
│   ├── contracts/
│   │   └── interfaces.py      # ImageProcessingInterface, ReceiptParsingInterface
│   └── image_processing.py    # VisionProcessor with DI support
│
├── tests/
│   ├── fakes/                 # Test doubles for DI
│   │   ├── fake_vision_processor.py
│   │   └── fake_receipt_parser.py
│   ├── integration/           # End-to-end tests with fakes
│   ├── domain/                # Domain layer tests
│   └── services/              # Service layer tests
└── imgs/                      # Place receipt images here
```

## Development Setup

The project uses a **src layout** to ensure clean imports and prevent module shadowing.

### Import Resolution

- **pytest** is configured via `pyproject.toml` with `testpaths = ["tests"]`
- **pip install -e .** installs the package in editable mode, enabling imports without PYTHONPATH
- Tests import from source modules directly: `from pipeline.processor import ReceiptProcessor`

### Running Tests

```bash
# Run all tests (imports resolve via src/)
pytest

# Or explicitly
python -m pytest tests/ -v

# Run specific test modules
pytest tests/pipeline_tests/ -v
pytest tests/integration/ -v
```

### Running the Application

```bash
# Run directly (after pip install -e .)
python main.py
```

### Project Configuration

All project configuration is centralized in `pyproject.toml`:
- pytest: `testpaths = ["tests"]` for test discovery
- Build system: setuptools with src layout (via `pip install -e .`)

TODO:
- [ ] Frontend app for image ingestion
- [ ] REST API for external integrations
- [ ] Horizontal scaling (cloud mode ready)
- [ ] Experiment with OCR models - PaddleOCR comparison