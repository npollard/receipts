"""Snapshot tests for receipt parsing.

Verifies parsed output matches expected structure from fixtures.
Uses structured comparison, not brittle string matching.
"""

import pytest
from pathlib import Path

from tests.snapshots import SnapshotHelper, normalize_receipt_for_snapshot
from tests.harness.pipeline_harness import PipelineTestHarness
from tests.harness.fakes import OCROutput
from tests.harness.fakes.fake_validation_service import ValidationField


class TestSimpleReceiptParsing:
    """Snapshot tests for simple grocery receipts."""

    def test_simple_receipt_matches_snapshot(self):
        """Given: Simple OCR text. When: Parsed. Then: Matches expected structure."""
        harness = PipelineTestHarness()
        helper = SnapshotHelper()

        harness.ocr.set_text_for_image(
            "receipt.jpg",
            "Grocery Store\n01/15/2024\nApples $3.99 x2\nBread $2.50\nMilk $4.00\nTotal: $25.99",
            quality=0.90
        )
        harness.parser.set_parse_result(
            merchant="Grocery Store",
            total=25.99,
            date="2024-01-15",
            items=[
                {"name": "Apples", "price": 3.99, "quantity": 2},
                {"name": "Bread", "price": 2.50, "quantity": 1},
                {"name": "Milk", "price": 4.00, "quantity": 1},
            ]
        )
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status.name == "SUCCESS"

        # Normalize and compare
        actual = normalize_receipt_for_snapshot(result.receipt_data or {})
        helper.assert_matches_snapshot(
            actual,
            "simple_receipt",
            ignore_order_fields=["items"]
        )

    def test_receipt_with_unordered_items(self):
        """Given: Items in different order. When: Parsed. Then: Still matches (order ignored)."""
        harness = PipelineTestHarness()
        helper = SnapshotHelper()

        harness.ocr.set_text_for_image("receipt.jpg", "STORE\nTotal: $25.99", quality=0.90)

        # Items in different order than snapshot
        harness.parser.set_parse_result(
            merchant="Grocery Store",
            total=25.99,
            date="2024-01-15",
            items=[
                {"name": "Milk", "price": 4.00, "quantity": 1},  # Last in expected
                {"name": "Apples", "price": 3.99, "quantity": 2},  # First in expected
                {"name": "Bread", "price": 2.50, "quantity": 1},
            ]
        )
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")
        actual = normalize_receipt_for_snapshot(result.receipt_data or {})

        # Should match despite different order (items field ignored for ordering)
        comparison = helper.compare(
            actual,
            "simple_receipt",
            ignore_order_fields=["items"]
        )
        assert comparison.matches, f"Order shouldn't matter: {comparison.differences}"


class TestRestaurantReceiptParsing:
    """Snapshot tests for restaurant receipts."""

    def test_restaurant_receipt_matches_snapshot(self):
        """Given: Restaurant receipt OCR. When: Parsed. Then: Matches expected."""
        harness = PipelineTestHarness()
        helper = SnapshotHelper()

        harness.ocr.set_text_for_image(
            "receipt.jpg",
            "Bistro Cafe\n02/20/2024\nCaesar Salad $12.95\nGrilled Salmon $24.95\nSparkling Water $3.50 x2\nTotal: $47.35",
            quality=0.90
        )
        harness.parser.set_parse_result(
            merchant="Bistro Cafe",
            total=47.35,
            date="2024-02-20",
            items=[
                {"name": "Caesar Salad", "price": 12.95, "quantity": 1},
                {"name": "Grilled Salmon", "price": 24.95, "quantity": 1},
                {"name": "Sparkling Water", "price": 3.50, "quantity": 2},
            ]
        )
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")
        actual = normalize_receipt_for_snapshot(result.receipt_data or {})

        helper.assert_matches_snapshot(
            actual,
            "restaurant_receipt",
            ignore_order_fields=["items"]
        )


class TestStandardReceiptParsing:
    """Snapshot tests for standard well-formed receipts."""

    def test_standard_receipt_matches_snapshot(self):
        """Given: Standard well-formed receipt. When: Parsed. Then: Matches expected structure."""
        harness = PipelineTestHarness()
        helper = SnapshotHelper()

        harness.ocr.set_text_for_image(
            "receipt.jpg",
            "Whole Foods Market\n03/15/2024\nOrganic Bananas $2.99\nWhole Milk $4.49\nSourdough Bread $5.99\nFree-Range Eggs $6.99\nSubtotal: $20.46\nTax: $1.74\nTotal: $22.20",
            quality=0.95
        )
        harness.parser.set_parse_result(
            merchant="Whole Foods Market",
            total=22.20,
            date="2024-03-15",
            items=[
                {"name": "Organic Bananas", "price": 2.99, "quantity": 1},
                {"name": "Whole Milk", "price": 4.49, "quantity": 1},
                {"name": "Sourdough Bread", "price": 5.99, "quantity": 1},
                {"name": "Free-Range Eggs", "price": 6.99, "quantity": 1},
            ],
            currency="USD",
            tax=1.74
        )
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        assert result.status.name == "SUCCESS"

        actual = normalize_receipt_for_snapshot(result.receipt_data or {})
        helper.assert_matches_snapshot(
            actual,
            "standard_receipt",
            ignore_order_fields=["items"]
        )


class TestNoisyOCRReceiptParsing:
    """Snapshot tests for noisy/poor quality OCR receipts."""

    def test_noisy_ocr_receipt_matches_snapshot(self):
        """Given: Noisy OCR output. When: Parsed with cleanup. Then: Matches expected."""
        harness = PipelineTestHarness()
        helper = SnapshotHelper()

        # Low quality OCR (simulating blurry image)
        harness.ocr.set_text_for_image(
            "receipt.jpg",
            "C0rn3r C@fe\n03/10/2024\nC0ffee $3.50\nB@gel $2.25\nTax $0.46\nTotal $6.21",
            quality=0.35  # Low quality triggers cleanup/fallback
        )
        harness.parser.set_parse_result(
            merchant="Corner Cafe",
            total=6.21,
            date="2024-03-10",
            items=[
                {"name": "Coffee", "price": 3.50, "quantity": 1},
                {"name": "Bagel", "price": 2.25, "quantity": 1},
            ],
            currency="USD",
            tax=0.46
        )
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        # Should succeed despite noisy OCR
        assert result.status.name == "SUCCESS"

        actual = normalize_receipt_for_snapshot(result.receipt_data or {})
        helper.assert_matches_snapshot(
            actual,
            "noisy_ocr_receipt",
            ignore_order_fields=["items"]
        )

    def test_noisy_ocr_with_fallback(self):
        """Given: Very noisy OCR. When: Fallback triggered. Then: Still produces valid result."""
        harness = PipelineTestHarness()

        # Very low quality - should trigger fallback
        harness.ocr.set_sequence([
            OCROutput(text="BLURRED", quality_score=0.15),  # Primary fails
            OCROutput(text="Corner Cafe\nCoffee $3.50", quality_score=0.75),  # Fallback succeeds
        ])

        harness.parser.set_parse_result(
            merchant="Corner Cafe",
            total=6.21,
            items=[
                {"name": "Coffee", "price": 3.50, "quantity": 1},
            ]
        )
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")

        # Should succeed via fallback
        assert result.status.name == "SUCCESS"
        assert result.ocr_method == "vision" or result.ocr_method is not None


class TestEdgeCaseReceiptParsing:
    """Snapshot tests for edge-case receipts with missing/incomplete fields."""

    def test_edge_case_missing_fields_matches_snapshot(self):
        """Given: Receipt with missing fields. When: Parsed. Then: Matches partial structure."""
        harness = PipelineTestHarness()
        helper = SnapshotHelper()

        harness.ocr.set_text_for_image(
            "receipt.jpg",
            "Unknown Vendor\n$15.00",  # Minimal info
            quality=0.60
        )
        harness.parser.set_parse_result(
            merchant="Unknown Vendor",
            total=15.00,
            date=None,
            items=[],
            currency=None,
            tax_amount=None
        )
        # Allow partial validation
        harness.validator.field_passes(ValidationField.MERCHANT)
        harness.validator.field_passes(ValidationField.TOTAL)
        harness.validator.set_preserve_partial(True)

        result = harness.run("receipt.jpg")

        # Should be partial success
        assert result.status.name in ["SUCCESS", "PARTIAL"]

        actual = normalize_receipt_for_snapshot(result.receipt_data or {})
        helper.assert_matches_snapshot(
            actual,
            "edge_case_receipt",
            ignore_order_fields=["items"]
        )

    def test_missing_date_and_currency(self):
        """Given: Missing optional fields. When: Parsed. Then: Nulls handled correctly."""
        harness = PipelineTestHarness()
        helper = SnapshotHelper()

        harness.ocr.set_text_for_image("receipt.jpg", "Vendor\nTotal: $15.00", quality=0.70)
        harness.parser.set_parse_result(
            merchant="Vendor",
            total=15.00,
            date=None,
            currency=None,
            items=[]
        )
        harness.validator.all_fields_pass()

        result = harness.run("receipt.jpg")
        actual = normalize_receipt_for_snapshot(result.receipt_data or {})

        # Should have nulls for missing fields
        assert actual.get("receipt_date") is None
        assert actual.get("currency") is None
        assert actual.get("total_amount") == 15.00


class TestSnapshotNormalization:
    """Tests for snapshot normalization helpers."""

    def test_float_normalization(self):
        """Given: Float values with many decimals. When: Normalized. Then: Rounded to 2 decimals."""
        data = {
            "total_amount": 25.999999,
            "items": [
                {"name": "Test", "price": 3.99999, "quantity": 1}
            ]
        }

        normalized = normalize_receipt_for_snapshot(data)

        assert normalized["total_amount"] == 26.00
        assert normalized["items"][0]["price"] == 4.00

    def test_item_sorting(self):
        """Given: Items in random order. When: Normalized. Then: Sorted by name."""
        data = {
            "items": [
                {"name": "Zebra", "price": 1.00},
                {"name": "Apple", "price": 2.00},
                {"name": "Mango", "price": 3.00},
            ]
        }

        normalized = normalize_receipt_for_snapshot(data)

        names = [item["name"] for item in normalized["items"]]
        assert names == ["Apple", "Mango", "Zebra"]

    def test_nested_normalization(self):
        """Given: Nested structures. When: Normalized. Then: All levels normalized."""
        data = {
            "subtotal": {
                "amount": 19.99999,
                "tax": 1.29999
            },
            "total": 21.29998
        }

        normalized = normalize_receipt_for_snapshot(data)

        assert normalized["subtotal"]["amount"] == 20.00
        assert normalized["subtotal"]["tax"] == 1.30
        assert normalized["total"] == 21.30


class TestSnapshotDifferenceDetection:
    """Tests for snapshot difference reporting."""

    def test_detects_missing_field(self):
        """Given: Actual missing expected field. When: Compared. Then: Reports difference."""
        helper = SnapshotHelper()

        actual = {"merchant_name": "Store", "total_amount": 10.00}
        # Missing "receipt_date" which is in snapshot

        comparison = helper.compare(actual, "simple_receipt")

        assert not comparison.matches
        assert any("receipt_date" in diff for diff in comparison.differences)

    def test_detects_type_mismatch(self):
        """Given: Type mismatch. When: Compared. Then: Reports difference."""
        helper = SnapshotHelper()

        # total_amount should be number, not string
        actual = {
            "merchant_name": "Grocery Store",
            "receipt_date": "2024-01-15",
            "total_amount": "25.99",  # String instead of number
        }

        comparison = helper.compare(actual, "simple_receipt", ignore_fields=["items"])

        assert not comparison.matches
        assert any("type mismatch" in diff for diff in comparison.differences)

    def test_detects_value_mismatch(self):
        """Given: Value differs. When: Compared. Then: Reports difference."""
        helper = SnapshotHelper()

        actual = {
            "merchant_name": "Wrong Store",  # Different from snapshot
            "receipt_date": "2024-01-15",
            "total_amount": 25.99,
        }

        comparison = helper.compare(actual, "simple_receipt", ignore_fields=["items"])

        assert not comparison.matches
        assert any("Wrong Store" in diff for diff in comparison.differences)

    def test_ignore_fields_works(self):
        """Given: Field in ignore list. When: Compared. Then: Difference ignored."""
        helper = SnapshotHelper()

        actual = {
            "merchant_name": "Grocery Store",
            "receipt_date": "2024-01-15",
            "total_amount": 25.99,
            "currency": "USD",
            "tax_amount": None,
            "items": [],  # Different from snapshot
            "extra_field": "should be ignored",
        }

        comparison = helper.compare(
            actual,
            "simple_receipt",
            ignore_fields=["extra_field", "items"]
        )

        # Should match despite extra field and items because they're ignored
        assert comparison.matches


class TestSnapshotUpdate:
    """Tests for updating snapshots."""

    def test_can_update_snapshot(self, tmp_path):
        """Given: New data. When: Snapshot updated. Then: Subsequent comparisons match."""
        # Use temp directory to avoid modifying real snapshots
        helper = SnapshotHelper(tmp_path)

        new_data = {
            "merchant_name": "New Store",
            "total_amount": 99.99,
            "items": []
        }

        # Update snapshot
        helper.update_snapshot(new_data, "new_snapshot")

        # Should now match
        comparison = helper.compare(new_data, "new_snapshot")
        assert comparison.matches

    def test_comparison_fails_after_data_changes(self, tmp_path):
        """Given: Changed data without update. When: Compared. Then: Fails."""
        helper = SnapshotHelper(tmp_path)

        original = {"value": 1}
        helper.update_snapshot(original, "test")

        changed = {"value": 2}
        comparison = helper.compare(changed, "test")

        assert not comparison.matches
