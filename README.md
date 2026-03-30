# Receipts

Leverage AI to take images of receipts and tell me where all my grocery money is going.

## LangChain Setup with Modular Architecture

This project uses LangChain for AI receipt parsing with a clean, modular architecture following DRY principles.

### Prerequisites

- Python 3.8+
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
- **`receipt_parser.py`** - Core AI parsing logic
- **`validation_utils.py`** - Validation and error handling
- **`token_utils.py`** - Token extraction utilities
- **`ai_parsing.py`** - Clean re-export of parsing functionality
- **`workflow.py`** - LangGraph workflow orchestration
- **`receipt_processor.py`** - Main processor facade
- **`main.py`** - CLI interface and entry point

**Key Classes:**
- **`ReceiptItem`** - Individual receipt items with description and price
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
export OPENAI_API_KEY="your-openai-api-key-here"

# Create directories
mkdir -p imgs
```

## Usage

### Basic Usage

```bash
# Process all images in imgs directory (shows persisted usage summary after processing)
python main.py

# Show only persisted token usage summary without processing
python main.py --usage-summary-only
```

### Environment Setup

```bash
# Required environment variables
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Configuration

The system uses LangChain-based processing with sequential OOP components and supports batch processing of all images in the `imgs/` directory.

### Batch Processing Features:
- **Automatic discovery** - Finds all .jpg, .jpeg, .png files in imgs/
- **Sequential processing** - Processes each image individually
- **Success tracking** - Reports success/failure rates
- **Token aggregation** - Tracks usage across all processed images

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

## Architecture Highlights

### DRY Principles
- **Single source of truth** - No duplicate parsing logic
- **Modular design** - Each module has clear responsibility
- **Reusable components** - Utils can be used across modules
- **Easy maintenance** - Changes localized to specific modules

### Data Flow
```
Images → VisionProcessor → OCR Text → ReceiptParser → Validation → Token Tracking → Storage
```

## Development

### Adding New Features

```python
# Add new validation logic to validation_utils.py
def custom_validation(data: dict) -> APIResponse:
    # Your validation logic here
    pass

# Extend token tracking in token_tracking.py
class CustomTokenUsage(TokenUsage):
    def custom_method(self):
        # Custom tracking logic
        pass
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
- **🔧 Maintainable** - Clean separation of concerns and DRY principles

## Recent Improvements

- **✅ Eliminated duplicate code** - Removed duplicate ai_parsing files
- **✅ Modular architecture** - Split into focused, single-responsibility modules
- **✅ Fixed validation issues** - Corrected Pydantic model field matching
- **✅ Improved error handling** - Better JSON serialization and fallbacks
- **✅ Enhanced logging** - Clear debugging and progress tracking
