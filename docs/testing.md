# Testing Guide

## Run All Tests

```bash
python -m pytest tests/ -v
```

## Integration Tests

```bash
# End-to-end tests with fakes (fast, no EasyOCR)
python -m pytest tests/integration/ -v
```

## Specific Test Modules

```bash
python -m pytest tests/test_api_response.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/services/ocr/test_ocr_scoring.py -v
```
