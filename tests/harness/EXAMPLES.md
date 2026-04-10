# Fake Usage Examples

## Basic OCR Fake

```python
from tests.harness.fakes import FakeOCRService

def test_ocr_extracts_text():
    ocr = FakeOCRService()
    
    # Configure fake to return specific text
    ocr.set_text_for_image("receipt.jpg", "GROCERY STORE\nTotal $25.50")
    
    # Use in test
    result = ocr.extract_text("receipt.jpg")
    assert result == "GROCERY STORE\nTotal $25.50"
    
    # Verify call was recorded
    assert ocr.get_call_count("extract_text") == 1
```

## OCR with Quality Score

```python
def test_ocr_quality_below_threshold_triggers_fallback():
    ocr = FakeOCRService()
    
    # First OCR attempt returns low quality
    ocr.set_text_for_image("receipt.jpg", "BLURRY...", quality=0.15)
    
    # Fallback returns high quality
    ocr.set_fallback_output("receipt.jpg", "GROCERY STORE\nTotal $25.50", quality=0.90)
    
    # Simulate threshold check
    text = ocr.extract_text("receipt.jpg")
    quality = ocr.score_ocr_quality(text)
    
    if quality < 0.25:
        text = ocr.extract_text("receipt.jpg", use_vision_fallback=True)
    
    assert ocr.used_fallback()
    assert "GROCERY STORE" in text
```

## OCR with Failure Simulation

```python
def test_ocr_failure_retries():
    from core.exceptions import OCRError
    
    ocr = FakeOCRService()
    
    # Fail on first attempt, succeed on second
    ocr.set_should_fail(
        OCRError("OCR failed"),
        max_attempts=1
    )
    ocr.set_text_for_image("receipt.jpg", "STORE $10")
    
    # First call raises
    with pytest.raises(OCRError):
        ocr.extract_text("receipt.jpg")
    
    # Second call succeeds
    ocr.reset_attempts()
    result = ocr.extract_text("receipt.jpg")
    assert result == "STORE $10"
```

## Parser Fake

```python
from tests.harness.fakes import FakeReceiptParser, ParserOutput

def test_parser_returns_structured_data():
    parser = FakeReceiptParser()
    
    # Configure success
    parser.set_parse_result(
        merchant="Grocery Store",
        total=25.50,
        date="2024-01-15",
        items=[
            {"description": "Milk", "price": 3.99},
            {"description": "Bread", "price": 2.50}
        ]
    )
    
    # Use in test
    response = parser.parse_text("OCR text here")
    
    assert response.status == "success"
    assert response.data["merchant_name"] == "Grocery Store"
    assert response.data["total_amount"] == 25.50
    assert len(response.data["items"]) == 2
```

## Parser with Malformed Output

```python
def test_parser_handles_invalid_json():
    parser = FakeReceiptParser()
    
    # Set raw JSON that will fail parsing
    parser.set_json_output("{invalid json}")
    
    # Or set explicit exception
    parser.set_parse_fails_with(json.JSONDecodeError("Invalid JSON", "", 0))
    
    with pytest.raises(json.JSONDecodeError):
        parser.parse_text("bad ocr")
```

## Parser with Retry Sequence

```python
def test_parser_retries_on_failure():
    parser = FakeReceiptParser()
    
    # First fails, second succeeds
    parser.set_parse_fails_with(ValueError("Missing date"))
    parser.set_parse_result(merchant="Store", total=10.0, date="2024-01-01")
    
    # First attempt
    with pytest.raises(ValueError):
        parser.parse_text("ocr text")
    
    # Second attempt
    result = parser.parse_text("ocr text")
    assert result.data["date"] == "2024-01-01"
    
    assert parser.get_call_count("parse_text") == 2
```

## Validation Fake

```python
from tests.harness.fakes import FakeValidationService, ValidationField

def test_validation_passes():
    validator = FakeValidationService()
    
    # Configure all fields to pass
    validator.all_fields_pass()
    
    result = validator.validate_receipt({
        "merchant": "Store",
        "total": 10.0
    })
    
    assert result.is_valid
    assert result.field_results["merchant"].is_valid
```

## Validation with Field Failures

```python
def test_validation_fails_on_missing_date():
    validator = FakeValidationService()
    
    # Most fields pass, but date fails
    validator.field_passes(ValidationField.MERCHANT)
    validator.field_fails(ValidationField.DATE, "Date is required")
    validator.field_passes(ValidationField.TOTAL)
    
    result = validator.validate_receipt({
        "merchant": "Store",
        "total": 10.0,
        "date": None
    })
    
    assert not result.is_valid
    assert "date" in result.errors[0].lower()
    assert validator.did_field_fail(ValidationField.DATE)
```

## Repository Fake

```python
from tests.harness.fakes import FakeRepository
from tests.harness.builders import ReceiptBuilder

def test_repository_saves_receipt():
    repo = FakeRepository()
    
    # Save a receipt
    receipt = repo.save_receipt(
        user_id="user_123",
        image_path="receipt.jpg",
        receipt_data={"merchant": "Store", "total": 10.0}
    )
    
    assert receipt.id is not None
    assert receipt.merchant_name == "Store"
    assert repo.get_save_count() == 1
    assert repo.was_saved_with_total(10.0)
```

## Repository with Idempotency

```python
def test_duplicate_receipt_detected():
    repo = FakeRepository()
    
    # Seed existing receipt
    existing = ReceiptBuilder().with_image_hash("abc123").build()
    repo.seed_receipt(existing)
    
    # Try to save same image again
    result = repo.save_receipt(
        user_id="user_123",
        image_path="receipt.jpg",
        receipt_data={"merchant": "Store", "total": 10.0},
        image_hash="abc123"  # Same hash
    )
    
    # Should return existing, not create new
    assert result.id == existing.id
    assert repo.duplicate_was_detected()
    assert repo.get_save_count() == 1  # Not 2
```

## Repository with Failure

```python
def test_repository_failure_handling():
    from core.exceptions import StorageError
    
    repo = FakeRepository()
    
    # Configure save to fail
    repo.set_should_fail_on("save", StorageError("DB connection failed"))
    
    with pytest.raises(StorageError):
        repo.save_receipt("user", "img.jpg", {})
```

## Retry Service Fake

```python
from tests.harness.fakes import FakeRetryService

def test_retry_succeeds_on_second_attempt():
    retry = FakeRetryService()
    
    # Succeed on 2nd attempt
    retry.set_succeed_on_attempt(2)
    
    call_count = 0
    def flaky_operation():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Temporary failure")
        return "success"
    
    result = retry.execute_with_retry(flaky_operation)
    
    assert result == "success"
    assert retry.get_attempt_count() == 2
    assert retry.was_retried()
    assert retry.succeeded_on_attempt(2)
```

## Retry with Strategy Selection

```python
def test_retry_uses_correct_strategy():
    from pydantic import ValidationError
    
    retry = FakeRetryService()
    
    # Different strategies for different errors
    retry.set_strategy_for_error(ValidationError, "LLM_SELF_CORRECTION")
    retry.set_strategy_for_error(ConnectionError, "EXPONENTIAL_BACKOFF")
    
    # Force success on first attempt for this test
    retry.set_succeed_on_attempt(1)
    
    # Track which strategy was used
    def operation_that_fails_validation():
        raise ValidationError("Invalid data")
    
    # In real usage, retry would catch and apply strategy
    # Here we just verify configuration
    result = retry.execute_with_retry(
        lambda: "success",  # Force success for demo
        error_types=(ValidationError,)
    )
    
    assert "LLM_SELF_CORRECTION" in retry.get_strategies_used()
```

## Using Builders

```python
from tests.harness.builders import (
    ReceiptBuilder,
    OCRResultBuilder,
    HarnessBuilder
)

# Receipt builder
def test_with_receipt_builder():
    receipt = (ReceiptBuilder()
        .with_merchant("Whole Foods")
        .with_total(45.67)
        .with_date("2024-03-15")
        .with_item("Organic Milk", 5.99)
        .with_item("Avocados", 3.49)
        .with_tax(2.50)
        .build())
    
    assert receipt.merchant_name == "Whole Foods"
    assert receipt.total_amount == 45.67

# Pre-configured receipts
def test_with_preconfigured():
    grocery = ReceiptBuilder.grocery_store(total=50.0).build()
    restaurant = ReceiptBuilder.restaurant(total=80.0).build()
    
    assert len(grocery.items) == 3
    assert restaurant.tax_amount == 8.0  # 10% tax

# OCR builder
def test_with_ocr_builder():
    ocr = (OCRResultBuilder()
        .with_text("SHOP\nItem $5.00")
        .with_high_quality()
        .using_easyocr()
        .with_processing_time(250)
        .build())
    
    assert ocr.quality_score == 0.9
    assert ocr.method == "easyocr"

# Harness builder - complete setup
def test_with_harness_builder():
    fakes = (HarnessBuilder()
        .with_ocr_text("receipt.jpg", "STORE $10", quality=0.8)
        .with_parsed_receipt(merchant="Store", total=10.0)
        .with_validation_passing()
        .with_retry_succeeding_first_time()
        .build())
    
    # Use fakes in test
    ocr = fakes["ocr"]
    parser = fakes["parser"]
    
    text = ocr.extract_text("receipt.jpg")
    result = parser.parse_text(text)
    
    assert result.data["total_amount"] == 10.0
```

## Full Pipeline Test Example

```python
def test_full_pipeline_with_fakes():
    """Example of testing the full pipeline with all fakes."""
    from pipeline.processor import ReceiptProcessor
    
    # Setup fakes
    ocr = FakeOCRService()
    ocr.set_text_for_image("receipt.jpg", "GROCERY $25.50")
    
    parser = FakeReceiptParser()
    parser.set_parse_result(merchant="Grocery", total=25.50)
    
    # Build processor with injected fakes
    processor = ReceiptProcessor(
        image_processor=ocr,
        receipt_parser=parser
    )
    
    # Execute pipeline
    result = processor.process_image("receipt.jpg")
    
    # Verify
    assert result.status == "success"
    assert ocr.get_call_count("extract_text") >= 1
    assert parser.get_call_count("parse_text") == 1
```
