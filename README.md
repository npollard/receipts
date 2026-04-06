# Receipts

AI-powered receipt processing to track grocery spending.

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

> Use a project virtual environment to avoid conflicts with globally installed LangChain packages.

```bash
# Clone the repository
git clone <repository-url>
cd receipts

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Set up environment
export OPENAI_API_KEY="your-openai-api-key-here"

# Create directory for receipt images
mkdir -p imgs
```

### VS Code

If you're using VS Code, select the workspace interpreter or let the included settings pick:

```bash
/Users/nelson/Development/receipts/.venv/bin/python
```

## Usage

```bash
# Process all images in imgs directory
python main.py

# Show token usage summary without processing
python main.py --usage-summary-only
```

Place receipt images (.jpg, .jpeg, .png) in the `imgs/` directory and run the processor.

## Testing

### Integration Tests

Run the integration tests with:

```bash
python3 -m pytest -q tests/test_integration_receipt_processing.py
```

Current integration coverage includes:
- Deterministic OCR stubs
- Deterministic LLM parsing stubs
- Mocked token usage validation
- Success and failure batch-processing scenarios

### Unit Tests

Run all unit tests with:

```bash
python3 -m pytest tests/ -v
```

Run specific test modules:

```bash
# Test API response handling
python3 -m pytest tests/test_api_response.py -v

# Test token usage tracking
python3 -m pytest tests/test_token_tracking.py -v

# Test data models
python3 -m pytest tests/test_models.py -v

# Test image processing
python3 -m pytest tests/test_image_processing.py -v

# Test receipt parsing
python3 -m pytest tests/test_receipt_parser.py -v

# Test workflow orchestration
python3 -m pytest tests/test_workflow.py -v

# Test validation utilities
python3 -m pytest tests/test_validation_utils.py -v

# Test receipt processor
python3 -m pytest tests/test_receipt_processor.py -v
```

Unit tests provide comprehensive coverage of:
- Individual component functionality
- Error handling and edge cases
- Data validation and model constraints
- Token usage tracking and cost estimation
- OCR and AI parsing logic with mocked dependencies

## Data Flow & Orchestration

```
Images → Vision API (OCR) → OCR Text Validation → AI Parser → Validation → Retry Logic → Storage
```

**Processing Pipeline:**
1. **Image Processing** - OpenAI Vision API extracts text from receipt images
2. **OCR Text Validation** - Validates extracted text quality and content
3. **AI Parsing** - GPT-4o-mini structures OCR text into JSON format
4. **Validation** - Pydantic models validate receipt structure and data types
5. **Retry Logic** - Up to 3 attempts with AI error-fixing for failed parses
6. **Token Tracking** - Monitors API usage and costs across all operations

**Error Recovery:**
- Automatic retries with targeted error-fixing prompts
- Fallback to empty structure if parsing repeatedly fails
- Detailed logging for debugging failed receipts

## Token Usage & Costs

The system tracks OpenAI API usage:
- **Input**: $0.15 per 1M tokens
- **Output**: $0.60 per 1M tokens
- Usage persists across sessions for cost analysis

## Output

Successfully parsed receipts include:
- Date, total amount, and itemized list
- Token usage statistics
- Processing status and any error details
