from decimal import Decimal

from domain.models.receipt import Receipt
from domain.parsing.receipt_parser import ParsingResult
from pipeline.duplicate_detector import DuplicateDetector
from pipeline.persistence_gateway import ReceiptPersistenceGateway
from pipeline.receipt_data_mapper import ReceiptDataMapper
from tracking import TokenUsage


def test_receipt_data_mapper_extracts_plain_data_from_parsing_result():
    parsed = ParsingResult(
        parsed=Receipt(
            date="2026-03-30",
            total=Decimal("7.50"),
            items=[{"description": "Milk", "price": Decimal("7.50")}],
        ),
        valid=True,
    )

    data = ReceiptDataMapper().extract(parsed)

    assert data == {
        "date": "2026-03-30",
        "items": [{"description": "Milk", "price": 7.5}],
        "total": 7.5,
    }


def test_duplicate_detector_prefers_precomputed_hash_lookup():
    class Repository:
        def __init__(self):
            self.seen_hash = None

        def find_by_image_hash(self, image_hash):
            self.seen_hash = image_hash
            return {"id": "existing"}

    repository = Repository()
    detector = DuplicateDetector(repository)

    assert detector.find_existing("receipt.jpg", "abc123") == {"id": "existing"}
    assert repository.seen_hash == "abc123"


def test_persistence_gateway_uses_fake_repository_shape():
    class Repository:
        user_id = "user-1"

        def save_receipt(self, user_id, image_path, receipt_data, image_hash):
            return {
                "id": "receipt-1",
                "user_id": user_id,
                "image_path": image_path,
                "receipt_data": receipt_data,
                "image_hash": image_hash,
            }

    saved = ReceiptPersistenceGateway(Repository()).save(
        receipt_data={"date": "2026-03-30", "total": 7.5, "items": []},
        image_path="receipt.jpg",
        image_hash="abc123",
        ocr_text="",
    )

    assert saved["id"] == "receipt-1"
    assert saved["user_id"] == "user-1"
    assert saved["image_hash"] == "abc123"


def test_persistence_gateway_uses_database_repository_shape():
    class Repository:
        def save_receipt(self, image_path, ocr_text, parsed_response, input_tokens=0, output_tokens=0):
            return {
                "id": "receipt-2",
                "image_path": image_path,
                "ocr_text": ocr_text,
                "status": parsed_response.status,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }, "created"

    usage = TokenUsage()
    usage.add_usage(10, 5)

    saved = ReceiptPersistenceGateway(Repository()).save(
        receipt_data={
            "date": "2026-03-30",
            "total": 7.5,
            "items": [{"description": "Milk", "price": 7.5}],
        },
        image_path="receipt.jpg",
        image_hash="abc123",
        ocr_text="OCR text",
        token_usage=usage,
    )

    assert ReceiptPersistenceGateway(Repository()).extract_id(saved) == "receipt-2"
    assert saved[0]["status"] == "success"
    assert saved[0]["input_tokens"] == 10
    assert saved[0]["output_tokens"] == 5
