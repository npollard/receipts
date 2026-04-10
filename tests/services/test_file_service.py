"""File handling service tests - pure unit tests for isolated file logic."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from services.file_service import FileHandlingService
from api_response import APIResponse


class TestFileValidation:
    """File validation scenarios."""

    def test_validate_existing_file(self, tmp_path):
        """Given: Existing file. When: Validated. Then: Returns True."""
        service = FileHandlingService()
        test_file = tmp_path / "test.jpg"
        test_file.write_text("fake image content")

        result = service.validate_file(test_file)

        assert result is True

    def test_validate_nonexistent_file(self, tmp_path):
        """Given: Non-existent file. When: Validated. Then: Returns False."""
        service = FileHandlingService()
        nonexistent = tmp_path / "missing.jpg"

        result = service.validate_file(nonexistent)

        assert result is False

    def test_validate_directory_not_file(self, tmp_path):
        """Given: Directory path. When: Validated. Then: Returns False."""
        service = FileHandlingService()
        directory = tmp_path / "subdir"
        directory.mkdir()

        result = service.validate_file(directory)

        assert result is False


class TestResultFormatting:
    """Result formatting scenarios."""

    def test_format_success_result(self, tmp_path):
        """Given: Success result. When: Formatted. Then: Contains all fields."""
        service = FileHandlingService()
        test_file = tmp_path / "receipt.jpg"
        test_file.write_text("content")

        result = APIResponse.success({"merchant": "Store", "total": 25.00})
        formatted = service.format_result(result, test_file)

        assert formatted["status"] == "success"
        assert formatted["data"]["merchant"] == "Store"
        assert formatted["file_path"] == str(test_file)
        assert formatted["file_name"] == "receipt.jpg"
        assert "file_size" in formatted

    def test_format_error_result(self, tmp_path):
        """Given: Error result. When: Formatted. Then: Contains error info."""
        service = FileHandlingService()
        test_file = tmp_path / "receipt.jpg"
        test_file.write_text("content")

        result = APIResponse.failure("Processing failed")
        formatted = service.format_result(result, test_file)

        assert formatted["status"] == "failed"
        assert "Processing failed" in formatted["error"]
        assert formatted["file_path"] == str(test_file)

    def test_format_result_with_missing_file(self, tmp_path):
        """Given: Missing file. When: Formatted. Then: File size is None."""
        service = FileHandlingService()
        missing_file = tmp_path / "missing.jpg"

        result = APIResponse.success({"data": "value"})
        formatted = service.format_result(result, missing_file)

        assert formatted["file_size"] is None


class TestResultEnrichment:
    """Result enrichment scenarios."""

    def test_enrich_with_metadata(self):
        """Given: Result and metadata. When: Enriched. Then: Metadata added."""
        service = FileHandlingService()
        result = APIResponse.success({"merchant": "Store"})
        metadata = {"processed_at": "2024-01-15", "version": "1.0"}

        enriched = service.enrich_result_with_metadata(result, metadata)

        assert enriched.data["merchant"] == "Store"
        assert enriched.data["processed_at"] == "2024-01-15"
        assert enriched.data["version"] == "1.0"

    def test_enrich_error_result(self):
        """Given: Error result. When: Enriched. Then: Returns enriched with error preserved."""
        service = FileHandlingService()
        result = APIResponse.failure("Original error")
        metadata = {"extra": "info"}

        enriched = service.enrich_result_with_metadata(result, metadata)

        # Error results get enriched too
        assert enriched.status == "failed"
        assert enriched.error == "Original error"

    def test_enrich_preserves_original(self):
        """Given: Result. When: Enriched. Then: Original result unchanged."""
        service = FileHandlingService()
        original = APIResponse.success({"original": "data"})
        metadata = {"new": "data"}

        enriched = service.enrich_result_with_metadata(original, metadata)

        # Enriched has both
        assert enriched.data["original"] == "data"
        assert enriched.data["new"] == "data"


class TestImageFileDiscovery:
    """Image file discovery scenarios."""

    def test_get_image_files_finds_images(self, tmp_path):
        """Given: Directory with images. When: Scanned. Then: Images found."""
        service = FileHandlingService()

        # Create test files
        (tmp_path / "image1.jpg").write_text("fake")
        (tmp_path / "image2.png").write_text("fake")
        (tmp_path / "not_image.txt").write_text("text")

        with patch('services.file_service.get_image_files') as mock_get:
            mock_get.return_value = [
                tmp_path / "image1.jpg",
                tmp_path / "image2.png"
            ]
            files = service.get_image_files(tmp_path)

        assert len(files) == 2
        assert all(f.suffix in ['.jpg', '.png'] for f in files)

    def test_validate_and_get_image_files_success(self, tmp_path):
        """Given: Directory with images. When: Validated and scanned. Then: Images returned."""
        service = FileHandlingService()

        (tmp_path / "receipt.jpg").write_text("fake")

        with patch('services.file_service.get_image_files') as mock_get:
            mock_get.return_value = [tmp_path / "receipt.jpg"]
            files = service.validate_and_get_image_files(tmp_path)

        assert len(files) == 1
        assert files[0].name == "receipt.jpg"

    def test_validate_and_get_image_files_missing_dir(self, tmp_path):
        """Given: Missing directory. When: Validated. Then: Empty list returned."""
        service = FileHandlingService()
        missing_dir = tmp_path / "nonexistent"

        files = service.validate_and_get_image_files(missing_dir)

        assert files == []

    def test_validate_and_get_image_files_no_images(self, tmp_path):
        """Given: Directory without images. When: Scanned. Then: Empty list returned."""
        service = FileHandlingService()
        (tmp_path / "readme.txt").write_text("text")

        with patch('services.file_service.get_image_files') as mock_get:
            mock_get.return_value = []
            files = service.validate_and_get_image_files(tmp_path)

        assert files == []


class TestDataConversion:
    """Data conversion scenarios."""

    def test_convert_dict_to_dict(self):
        """Given: Dict data. When: Converted. Then: Returns same dict."""
        service = FileHandlingService()
        data = {"key": "value", "nested": {"a": 1}}

        result = service._convert_data_to_dict(data)

        assert result == data
        assert result is data  # Same object

    def test_convert_none_to_empty_dict(self):
        """Given: None. When: Converted. Then: Returns empty dict."""
        service = FileHandlingService()

        result = service._convert_data_to_dict(None)

        assert result == {}

    def test_convert_pydantic_model(self):
        """Given: Pydantic model. When: Converted. Then: Dict returned."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        service = FileHandlingService()
        model = TestModel(name="test", value=42)

        result = service._convert_data_to_dict(model)

        assert result == {"name": "test", "value": 42}

    def test_format_result_data(self):
        """Given: APIResponse. When: Data formatted. Then: Dict returned."""
        service = FileHandlingService()
        result = APIResponse.success({"data": "value", "number": 123})

        formatted = service.format_result_data(result)

        assert formatted == {"data": "value", "number": 123}


class TestResultProcessing:
    """Result processing scenarios."""

    def test_process_single_image_result_success(self, tmp_path, caplog):
        """Given: Success result. When: Processed. Then: Logged appropriately."""
        service = FileHandlingService()
        test_file = tmp_path / "receipt.jpg"
        test_file.write_text("fake")

        result = APIResponse.success({
            "extracted_text": "STORE $25.00",
            "parsed_receipt": {"merchant": "Store"}
        })

        with patch.object(service.logger, 'info') as mock_info:
            service.process_single_image_result(result, test_file)

            # Should log success
            assert mock_info.called

    def test_process_single_image_result_failure(self, tmp_path):
        """Given: Failure result. When: Processed. Then: Error logged."""
        service = FileHandlingService()
        test_file = tmp_path / "receipt.jpg"
        test_file.write_text("fake")

        result = APIResponse.failure("Processing failed")

        with patch.object(service.logger, 'error') as mock_error:
            service.process_single_image_result(result, test_file)

            assert mock_error.called


class TestServiceInterface:
    """Service interface compliance scenarios."""

    def test_implements_file_handling_interface(self):
        """Given: FileHandlingService. When: Checked. Then: Implements interface."""
        from contracts.interfaces import FileHandlingInterface

        service = FileHandlingService()

        # Should be instance of interface
        assert isinstance(service, FileHandlingInterface)

    def test_required_methods_exist(self):
        """Given: Service. When: Checked. Then: Has all required methods."""
        service = FileHandlingService()

        required_methods = [
            'validate_file',
            'format_result',
            'get_image_files',
            'enrich_result_with_metadata',
            'validate_and_get_image_files',
            'format_result_data',
        ]

        for method in required_methods:
            assert hasattr(service, method), f"Missing method: {method}"
            assert callable(getattr(service, method)), f"Method not callable: {method}"


class TestEdgeCases:
    """Edge case scenarios."""

    def test_format_result_with_empty_data(self, tmp_path):
        """Given: Success with empty data. When: Formatted. Then: Handles gracefully."""
        service = FileHandlingService()
        test_file = tmp_path / "receipt.jpg"
        test_file.write_text("fake")

        result = APIResponse.success({})
        formatted = service.format_result(result, test_file)

        assert formatted["status"] == "success"
        assert formatted["data"] == {}

    def test_enrich_with_empty_metadata(self):
        """Given: Empty metadata. When: Enriched. Then: Original unchanged."""
        service = FileHandlingService()
        result = APIResponse.success({"original": "data"})

        enriched = service.enrich_result_with_metadata(result, {})

        assert enriched.data == {"original": "data"}

    def test_convert_data_with_nested_model(self):
        """Given: Nested Pydantic models. When: Converted. Then: Recursive conversion."""
        from pydantic import BaseModel
        from typing import Optional

        class Nested(BaseModel):
            value: int

        class Parent(BaseModel):
            name: str
            child: Optional[Nested] = None

        service = FileHandlingService()
        model = Parent(name="parent", child=Nested(value=42))

        result = service._convert_data_to_dict(model)

        assert result == {"name": "parent", "child": {"value": 42}}
