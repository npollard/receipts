from pathlib import Path
from api_response import APIResponse
from pipeline.processor import ReceiptProcessor
from services.batch_service import BatchProcessingService
from image_processing import VisionProcessor
from domain.parsing.receipt_parser import ReceiptParser


def _build_receipt(date: str, items: list[dict], total: float, input_tokens: int, output_tokens: int) -> dict:
    return {
        "date": date,
        "items": items,
        "total": total,
        "_token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def test_receipt_processor_process_directly_returns_expected_receipt_and_token_usage(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"fake-image-bytes")

    expected_ocr = "MILK 4.50\nBREAD 3.00\nTOTAL 7.50"
    expected_receipt = _build_receipt(
        date="2026-03-30",
        items=[
            {"description": "Milk", "price": 4.50},
            {"description": "Bread", "price": 3.00},
        ],
        total=7.50,
        input_tokens=120,
        output_tokens=45,
    )

    def fake_extract_text(self, path: str) -> str:
        assert path == str(image_path)
        return expected_ocr

    def fake_parse_text(self, text: str):
        assert text == expected_ocr
        from domain.parsing.receipt_parser import ParsingResult
        result = ParsingResult()
        result.parsed = {k: v for k, v in expected_receipt.items() if k != "_token_usage"}
        result.valid = True
        result.error = None
        result.token_usage.add_usage(
            expected_receipt["_token_usage"]["input_tokens"],
            expected_receipt["_token_usage"]["output_tokens"],
        )
        return result

    def fake_extract_text_with_obs(self, path: str):
        from services.ocr_service import OCRObservability
        text = fake_extract_text(self, path)
        obs = OCRObservability(
            method="test",
            start_time=0.0,
            end_time=1.0,
            quality_score=0.95,
            text_length=len(text),
            confidence_threshold=0.5,
            attempted_methods=["test"]
        )
        return text, obs

    monkeypatch.setattr("services.ocr_service.OCRService.extract_text_with_observability", fake_extract_text_with_obs)
    monkeypatch.setattr(
        "domain.parsing.receipt_parser.ReceiptParser.parse_text",
        fake_parse_text,
    )

    processor = ReceiptProcessor()
    result = processor.process_directly(str(image_path))

    assert result.status == "success"
    assert result.data == {
        "image_path": str(image_path),
        "ocr_text": expected_ocr,
        "parsed_receipt": expected_receipt,
    }
    assert processor.token_usage.to_dict() == {
        "input_tokens": 120,
        "output_tokens": 45,
        "total_tokens": 165,
        "estimated_cost": 0.045,
        "model_name": "gpt-4o-mini",
    }


def test_process_batch_images_aggregates_stubbed_token_usage(monkeypatch, tmp_path: Path):
    image_paths = [tmp_path / "receipt1.png", tmp_path / "receipt2.png"]
    for image_path in image_paths:
        image_path.write_bytes(b"fake-image-bytes")

    expected_ocr_by_path = {
        str(image_paths[0]): "COFFEE 3.25\nMUFFIN 2.75\nTOTAL 6.00",
        str(image_paths[1]): "TEA 2.50\nCOOKIE 1.50\nTOTAL 4.00",
    }
    receipt_by_ocr = {
        expected_ocr_by_path[str(image_paths[0])]: _build_receipt(
            date="2026-03-30",
            items=[
                {"description": "Coffee", "price": 3.25},
                {"description": "Muffin", "price": 2.75},
            ],
            total=6.00,
            input_tokens=80,
            output_tokens=20,
        ),
        expected_ocr_by_path[str(image_paths[1])]: _build_receipt(
            date="2026-03-30",
            items=[
                {"description": "Tea", "price": 2.50},
                {"description": "Cookie", "price": 1.50},
            ],
            total=4.00,
            input_tokens=70,
            output_tokens=15,
        ),
    }

    def fake_extract_text(self, path: str) -> str:
        return expected_ocr_by_path[path]

    def fake_parse_with_retry(self, ocr_text: str, image_path: str = None):
        from domain.parsing.receipt_parser import ParsingResult
        receipt_data = receipt_by_ocr[ocr_text]
        result = ParsingResult()
        result.parsed = {k: v for k, v in receipt_data.items() if k != "_token_usage"}
        result.valid = True
        result.error = None
        result.token_usage.add_usage(
            receipt_data["_token_usage"]["input_tokens"],
            receipt_data["_token_usage"]["output_tokens"],
        )
        return result

    monkeypatch.setattr("image_processing.VisionProcessor.extract_text", fake_extract_text)
    monkeypatch.setattr(
        "domain.parsing.receipt_parser.ReceiptParser.parse_with_validation_driven_retry",
        fake_parse_with_retry,
    )

    # Use BatchProcessingService directly instead of main.process_batch_images
    image_processor = VisionProcessor()
    receipt_parser = ReceiptParser()
    batch_service = BatchProcessingService()
    successful, failed, token_usage, _ = batch_service.process_batch(
        image_paths, image_processor, receipt_parser
    )

    assert (successful, failed) == (2, 0)
    assert token_usage.to_dict() == {
        "input_tokens": 150,
        "output_tokens": 35,
        "total_tokens": 185,
        "estimated_cost": 0.0435,
        "model_name": "gpt-4o-mini",
    }


def test_process_batch_images_tracks_only_successful_token_usage_when_one_parse_fails(monkeypatch, tmp_path: Path):
    image_paths = [tmp_path / "receipt-ok.png", tmp_path / "receipt-failed.png"]
    for image_path in image_paths:
        image_path.write_bytes(b"fake-image-bytes")

    successful_ocr = "BANANA 1.25\nYOGURT 2.75\nTOTAL 4.00"
    failed_ocr = "UNREADABLE RECEIPT CONTENT"
    expected_ocr_by_path = {
        str(image_paths[0]): successful_ocr,
        str(image_paths[1]): failed_ocr,
    }
    successful_receipt = _build_receipt(
        date="2026-03-30",
        items=[
            {"description": "Banana", "price": 1.25},
            {"description": "Yogurt", "price": 2.75},
        ],
        total=4.00,
        input_tokens=50,
        output_tokens=10,
    )

    def fake_extract_text(self, path: str) -> str:
        return expected_ocr_by_path[path]

    def fake_parse_with_retry(self, ocr_text: str, image_path: str = None):
        from domain.parsing.receipt_parser import ParsingResult
        if ocr_text == successful_ocr:
            result = ParsingResult()
            result.parsed = {k: v for k, v in successful_receipt.items() if k != "_token_usage"}
            result.valid = True
            result.error = None
            result.token_usage.add_usage(
                successful_receipt["_token_usage"]["input_tokens"],
                successful_receipt["_token_usage"]["output_tokens"],
            )
            return result
        # Return failure result
        result = ParsingResult()
        result.parsed = None
        result.valid = False
        result.error = "Validation failed: missing total"
        return result

    monkeypatch.setattr("image_processing.VisionProcessor.extract_text", fake_extract_text)
    monkeypatch.setattr(
        "domain.parsing.receipt_parser.ReceiptParser.parse_with_validation_driven_retry",
        fake_parse_with_retry,
    )

    # Use BatchProcessingService directly instead of main.process_batch_images
    image_processor = VisionProcessor()
    receipt_parser = ReceiptParser()
    batch_service = BatchProcessingService()
    successful, failed, token_usage, _ = batch_service.process_batch(
        image_paths, image_processor, receipt_parser
    )

    assert (successful, failed) == (1, 1)
    assert token_usage.to_dict() == {
        "input_tokens": 50,
        "output_tokens": 10,
        "total_tokens": 60,
        "estimated_cost": 0.0135,
        "model_name": "gpt-4o-mini",
    }
