from dataclasses import dataclass

from domain.models.receipt import Receipt
from domain.parsing.receipt_parser import ParsingResult
from pipeline.processor import PipelineStatus, Processor
from tests.harness.fakes.fake_repository import FakeRepository
from tracking import TokenUsage


@dataclass
class _Obs:
    method: str = "fake"
    quality_score: float = 0.9


class _OCR:
    def __init__(self):
        self.calls = 0

    def extract_text_with_observability(self, image_path):
        self.calls += 1
        return "STORE\nTOTAL 7.50", _Obs()


class _Parser:
    def parse_text(self, text):
        usage = TokenUsage()
        usage.add_usage(10, 5)
        return ParsingResult(
            parsed=Receipt(
                date="2026-03-30",
                total=7.50,
                items=[{"description": "Milk", "price": 7.50}],
            ),
            valid=True,
            token_usage=usage,
        )


class _Validation:
    is_valid = True
    retryable = False
    preserve_partial = False


class _Validator:
    def validate_receipt(self, parsed):
        return _Validation()


def test_processor_uses_image_hash_for_duplicate_lookup(tmp_path):
    image_path = tmp_path / "receipt.jpg"
    image_path.write_bytes(b"same-image-bytes")
    ocr = _OCR()
    repository = FakeRepository()
    processor = Processor(
        ocr_service=ocr,
        parser=_Parser(),
        validator=_Validator(),
        repository=repository,
    )

    first = processor.process(str(image_path))
    second = processor.process(str(image_path))

    assert first.status == PipelineStatus.SUCCESS
    assert second.status == PipelineStatus.DUPLICATED
    assert second.was_duplicate is True
    assert ocr.calls == 1
    assert repository.get_metrics().actual_writes == 1
