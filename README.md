# Receipts

Leverage AI to take images of receipts and tell me where all my grocery money is going.

## LangChain/LangGraph Setup with Modular Architecture

This project uses LangChain/LangGraph for workflow management with modular OOP architecture.

### Prerequisites

- Python 2.7.4+
- OpenAI API key
- Tesseract OCR

### Setup

The project is split into modular components:

**Core Modules:**
- **`models.py`** - Pydantic data models (Receipt, ReceiptItem)
- **`api_response.py`** - Standardized API response structure
- **`token_tracking.py`** - Token usage tracking utilities
- **`token_usage_persistence.py`** - Long-term token usage storage
- **`image_processing.py`** - OpenAI Vision API integration
- **`ai_parsing_with_persistence.py`** - AI parsing with persistence
- **`workflow.py`** - LangGraph workflow orchestration
- **`receipt_processor.py`** - Main processor facade
- **`main.py`** - CLI interface and entry point

**Key Classes:**
- **`ReceiptItem`** - Individual receipt items with validation
- **`Receipt`** - Complete receipt with total validation
- **`VisionProcessor`** - Direct image-to-text processing
- **`ReceiptParser`** - AI parsing with retry and error fixing
- **`WorkflowOrchestrator`** - LangGraph-based workflow management
- **`TokenUsagePersistence`** - Persistent token usage storage

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd receipts

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your OpenAI API key to .env

# Create directories
mkdir -p imgs
```

## Usage

### Basic Usage

```bash
# Process a single image
python main.py --mode direct imgs/receipt.jpg

# Process with LangGraph workflow
python main.py --mode langgraph imgs/receipt.jpg

# Process with LangChain chains
python main.py --mode chain imgs/receipt.jpg
```

### Token Usage Management

```bash
# View token usage summary
python main.py --usage-summary

# Clear old sessions (keep last 50)
python main.py --clear-sessions --keep-last 50
```

### Environment Setup

```bash
# Required environment variables
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Configuration

The system supports multiple processing modes:

- **Direct Mode**: Simple image → OCR → AI parsing
- **LangGraph Mode**: Workflow-based processing with state management
- **Chain Mode**: LangChain sequential processing

## Token Usage & Cost Tracking

The system automatically tracks:
- **Input/Output tokens** per parsing session
- **Estimated costs** based on GPT-4o-mini pricing
- **Session history** with timestamps and usage data
- **Persistent storage** in JSON format for long-term analysis

### Pricing Model (GPT-4o-mini)

- **Input**: $0.15 per 1M tokens
- **Output**: $0.60 per 1M tokens
- **Example**: 1000 input + 500 output = $0.15 + $0.30 = $0.45

## Error Handling

- **Automatic retries** - Up to 3 attempts with AI error fixing
- **Validation errors** - Detailed error reporting with JSON serialization
- **Fallback mechanisms** - Graceful degradation when parsing fails
- **Comprehensive logging** - Detailed debugging information

## Development

### Adding New Processing Modes

```python
# Add new mode to workflow.py
def process_image_with_new_mode(image_path: str) -> APIResponse:
    # Your new processing logic here
    pass

# Update main.py to include new mode
parser.add_argument('--mode', choices=['direct', 'langgraph', 'chain', 'new_mode'], default='direct')
```

### Extending Token Usage

```python
# Add custom cost calculation
from token_usage_persistence import TokenUsagePersistence

persistence = TokenUsagePersistence("custom_storage.json")
# Custom usage tracking logic
```

## Performance

- **🚀 Fast processing** - Direct Vision API without intermediate OCR steps
- **🎯 High accuracy** - GPT-4o-mini understands receipt context
- **💰 Cost effective** - Only pay for successful parsing operations
- **📈 Scalable** - Modular architecture supports easy extension
