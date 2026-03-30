# Receipts

AI-powered receipt processing to track grocery spending.

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd receipts

# Install dependencies
pip install -r requirements.txt

# Set up environment
export OPENAI_API_KEY="your-openai-api-key-here"

# Create directory for receipt images
mkdir -p imgs
```

## Usage

```bash
# Process all images in imgs directory
python main.py

# Show token usage summary without processing
python main.py --usage-summary-only
```

Place receipt images (.jpg, .jpeg, .png) in the `imgs/` directory and run the processor.

## Data Flow & Orchestration

```
Images → Vision API (OCR) → AI Parser → Validation → Storage
```

**Processing Pipeline:**
1. **Image Processing** - OpenAI Vision API extracts text from receipt images
2. **AI Parsing** - GPT-4o-mini structures OCR text into JSON format
3. **Validation** - Pydantic models validate receipt structure and data types
4. **Retry Logic** - Up to 3 attempts with AI error-fixing for failed parses
5. **Token Tracking** - Monitors API usage and costs across all operations

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
