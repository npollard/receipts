import inspect
from decimal import Decimal
from typing import Any, Optional

from api_response import APIResponse
from domain.models.receipt import Receipt
from services.retry_service import RetryService
from tracking.usage import TokenUsage


class PipelineStatus:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"
    DUPLICATED = "DUPLICATED"


class PipelineResult:
    def __init__(self):
        self.status = None
        self.receipt_id: Optional[str] = None
        self.retry_count: int = 0
        self.data: Optional[dict] = None
        self.was_duplicate: bool = False
        self.receipt_data: Optional[dict] = None


class TraceEntry:
    def __init__(self, stage: str, success: bool, data: Optional[dict] = None):
        self.stage = stage
        self.success = success
        self.data = data or {}


class InMemoryRepository:
    """Simple in-memory repository for default Processor construction."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self._storage = {}
        self._image_hash_index = {}

    def find_by_hash(self, image_hash: str) -> Optional[Any]:
        """Check if receipt exists by image hash."""
        return self._image_hash_index.get(image_hash)

    def save(self, data: dict) -> dict:
        """Save receipt data and return with generated id."""
        import uuid
        receipt_id = str(uuid.uuid4())
        result = dict(data)
        result["id"] = receipt_id
        self._storage[receipt_id] = result
        return result

    def save_receipt(self, user_id: str, image_path: str, receipt_data: dict, image_hash: str) -> Any:
        """Save receipt with all metadata."""
        import uuid
        from dataclasses import dataclass

        @dataclass
        class ReceiptDTO:
            id: str
            user_id: str
            image_path: str
            receipt_data: dict

        receipt_id = str(uuid.uuid4())
        dto = ReceiptDTO(
            id=receipt_id,
            user_id=user_id,
            image_path=image_path,
            receipt_data=receipt_data
        )
        self._storage[receipt_id] = dto
        self._image_hash_index[image_hash] = dto
        return dto


class Processor:
    def __init__(
        self,
        ocr_service=None,
        parser=None,
        validator=None,
        repository=None,
        retry_service=None,
        image_processor=None,
        receipt_parser=None,
        db_manager=None
    ):
        # Map legacy parameter names for backwards compatibility
        if ocr_service is None and image_processor is not None:
            ocr_service = image_processor
        if parser is None and receipt_parser is not None:
            parser = receipt_parser
        if repository is None and db_manager is not None:
            repository = db_manager

        # Initialize defaults for any None dependencies
        if ocr_service is None:
            from services.ocr_service import OCRService
            ocr_service = OCRService()

        if parser is None:
            from domain.parsing.receipt_parser import ReceiptParser
            parser = ReceiptParser()

        if validator is None:
            from domain.validation.validation_service import ValidationService
            validator = ValidationService()

        if repository is None:
            # Use in-memory repository for default construction (no DB required)
            repository = InMemoryRepository(user_id="default-user")

        self.ocr_service = ocr_service
        self.parser = parser
        self.validator = validator
        self.repository = repository
        self.retry_service = retry_service or RetryService(max_retries=3)
        self.token_usage = TokenUsage()

        self.trace = []

        # User context for backwards compatibility
        self._user_context = {"email": "default@example.com", "user_id": "default"}

    # -----------------------------
    # User Management (backwards compatibility)
    # -----------------------------

    def switch_user(self, email: str):
        """Switch to a different user context."""
        self._user_context["email"] = email
        # Extract user_id from email or use email as user_id
        self._user_context["user_id"] = email
        # Update repository user_id if it has one
        if hasattr(self.repository, 'user_id'):
            self.repository.user_id = email
        # Also try setter method if available
        if hasattr(self.repository, 'set_user_id'):
            self.repository.set_user_id(email)

    def get_user_context(self) -> dict:
        """Get current user context."""
        # If repository has user context, use that
        if hasattr(self.repository, 'user_id'):
            return {
                "email": self.repository.user_id,
                "user_id": self.repository.user_id
            }
        return self._user_context

    # -----------------------------
    # Public API
    # -----------------------------

    def process(self, image_path: str) -> PipelineResult:
        result = PipelineResult()
        total_retries = 0
        self._last_image_path = image_path

        # -----------------------------
        # DUPLICATE DETECTION
        # -----------------------------
        image_hash = self._compute_hash(image_path)
        self._last_image_hash = image_hash
        existing = self._find_existing_receipt(image_path, image_hash)

        if existing:
            result.status = PipelineStatus.DUPLICATED
            result.was_duplicate = True
            result.receipt_id = existing.id if hasattr(existing, "id") else existing.get("id")
            result.receipt_data = existing if isinstance(existing, dict) else getattr(existing, "__dict__", None)
            return result

        # -----------------------------
        # OCR (bounded attempts)
        # -----------------------------
        ocr_text = self._run_ocr(image_path)
        # Store for access by process_directly
        self._last_ocr_text = ocr_text

        if ocr_text is None:
            result.status = PipelineStatus.FAILURE
            result.receipt_id = None
            return result

        # -----------------------------
        # PARSE (retry)
        # -----------------------------
        self.retry_service._reset_tracking()
        try:
            parsed = self.retry_service.execute_with_retry(
                self.parser.parse_text,
                ocr_text,
                error_types=(Exception,),
            )
            self._record_trace("PARSE", True)
            # Copy token usage from ParsingResult to processor
            if hasattr(parsed, 'token_usage') and parsed.token_usage:
                self.token_usage = parsed.token_usage

        except Exception as e:
            self._record_trace("PARSE", False)
            result.status = PipelineStatus.FAILURE
            result.receipt_id = None
            return result

        attempts = self.retry_service.get_attempt_count()
        total_retries += max(0, attempts - 1)

        # -----------------------------
        # VALIDATE
        # -----------------------------
        # Handle both validate() and validate_receipt() interfaces
        if hasattr(self.validator, 'validate_receipt'):
            validation = self.validator.validate_receipt(parsed)
        elif hasattr(self.validator, 'validate'):
            validation = self.validator.validate(parsed)
        else:
            # Fallback: assume valid if no validator method found
            class FakeValidation:
                is_valid = True
                retryable = False
                preserve_partial = False
            validation = FakeValidation()
        self._record_trace("VALIDATE", validation.is_valid)

        if not validation.is_valid:

            # Retryable → one controlled retry
            if getattr(validation, "retryable", False):
                self.retry_service._reset_tracking()
                try:
                    parsed = self.retry_service.execute_with_retry(
                        self.parser.parse_text,
                        ocr_text,
                        error_types=(Exception,),
                    )
                    attempts = self.retry_service.get_attempt_count()
                    total_retries += max(0, attempts - 1)

                except Exception:
                    result.status = PipelineStatus.FAILURE
                    result.receipt_id = None
                    return result

                validation = self.validator.validate(parsed)

            # Final resolution
            if not validation.is_valid:
                if getattr(validation, "preserve_partial", False):
                    return self._persist(parsed, result, total_retries, partial=True)
                else:
                    result.status = PipelineStatus.FAILURE
                    result.receipt_id = None
                    return result

        # -----------------------------
        # PERSIST
        # -----------------------------
        return self._persist(parsed, result, total_retries)

    def process_directly(self, image_path: str):
        """
        Backwards compatibility method used by integration tests.
        """
        result = self.process(image_path)

        # Create return object with expected shape for integration tests
        class DirectResult:
            def __init__(self, status, data):
                self.status = status
                self.data = data

        # Build status string
        status = "success" if result.status == PipelineStatus.SUCCESS else result.status.lower()

        # Extract receipt data from ParsingResult if needed
        receipt_data = result.receipt_data if result.receipt_data else {}
        if hasattr(receipt_data, 'parsed'):
            parsed_receipt = dict(receipt_data.parsed)  # Make a copy
            # Add token_usage from ParsingResult if available
            if hasattr(receipt_data, 'token_usage'):
                tu = receipt_data.token_usage
                parsed_receipt['_token_usage'] = {
                    'input_tokens': tu.input_tokens,
                    'output_tokens': tu.output_tokens,
                    'total_tokens': tu.input_tokens + tu.output_tokens,
                }
        elif hasattr(receipt_data, 'receipt_data'):
            parsed_receipt = receipt_data.receipt_data
        else:
            parsed_receipt = receipt_data

        # Get OCR text from stored value
        ocr_text = getattr(self, '_last_ocr_text', '')

        # Build data dict with expected structure
        data = {
            "image_path": image_path,
            "ocr_text": ocr_text,
            "parsed_receipt": parsed_receipt,
        }

        return DirectResult(status, data)

    # -----------------------------
    # OCR
    # -----------------------------


    def _run_ocr(self, image_path: str) -> Optional[str]:
        max_attempts = 3

        for _ in range(max_attempts):
            try:
                text, obs = self.ocr_service.extract_text_with_observability(image_path)

                success = bool(text and text.strip())

                self._record_trace(
                    "OCR",
                    success,
                    data={"method": obs.method, "quality": obs.quality_score}
                )

                if success:
                    return text

            except Exception:
                self._record_trace("OCR", False)

        return None

    # -----------------------------
    # Persistence
    # -----------------------------

    def _persist(self, parsed, result: PipelineResult, retry_count: int, partial: bool = False):
        receipt_id = None

        self.retry_service._reset_tracking()
        try:
            receipt_data = self._extract_receipt_data(parsed)

            if hasattr(self.repository, 'save_receipt'):
                saved = self.retry_service.execute_with_retry(
                    self._save_receipt,
                    receipt_data,
                    error_types=(Exception,),
                )
            elif hasattr(self.repository, 'save'):
                saved = self.retry_service.execute_with_retry(
                    self.repository.save,
                    receipt_data,
                    error_types=(Exception,),
                )
            else:
                raise AttributeError("Repository has no save method")

            receipt_id = self._extract_id(saved)
            result.data = parsed
            result.receipt_data = parsed
            result.status = PipelineStatus.PARTIAL if partial else PipelineStatus.SUCCESS

            self._record_trace("PERSIST", True)

        except Exception:
            self._record_trace("PERSIST", False)
            result.status = PipelineStatus.FAILURE
            result.data = parsed
            result.receipt_data = parsed

        result.receipt_id = receipt_id

        attempts = self.retry_service.get_attempt_count()
        total_retries = retry_count + max(0, attempts - 1)
        result.retry_count = total_retries

        return result

    # -----------------------------
    # Helpers
    # -----------------------------

    def _extract_id(self, saved: Any) -> Optional[str]:
        if saved is None:
            return None
        if isinstance(saved, tuple) and saved:
            return self._extract_id(saved[0])
        if isinstance(saved, dict):
            return saved.get("id")
        return getattr(saved, "id", None)

    def _record_trace(self, stage: str, success: bool, data: Optional[dict] = None):
        self.trace.append(TraceEntry(stage, success, data))

    def _compute_hash(self, image_path: str) -> str:
        from core.hashing import calculate_image_hash
        return calculate_image_hash(image_path)

    def _find_existing_receipt(self, image_path: str, image_hash: str) -> Optional[Any]:
        if hasattr(self.repository, "find_by_image_hash"):
            return self.repository.find_by_image_hash(image_hash)
        if hasattr(self.repository, "find_by_hash"):
            return self.repository.find_by_hash(image_hash)
        if hasattr(self.repository, "find_existing_receipt_by_image_hash"):
            return self.repository.find_existing_receipt_by_image_hash(image_path)
        return None

    def _extract_receipt_data(self, parsed: Any) -> Any:
        if hasattr(parsed, 'parsed'):
            receipt_data = parsed.parsed
        elif hasattr(parsed, 'receipt_data'):
            receipt_data = parsed.receipt_data
        else:
            receipt_data = parsed

        if hasattr(receipt_data, "model_dump"):
            receipt_data = receipt_data.model_dump()
        return self._to_plain_data(receipt_data)

    def _to_plain_data(self, value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, dict):
            return {key: self._to_plain_data(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_plain_data(item) for item in value]
        return value

    def _save_receipt(self, receipt_data: Any) -> Any:
        save_receipt = self.repository.save_receipt
        params = inspect.signature(save_receipt).parameters
        image_path = getattr(self, "_last_image_path", "unknown")
        image_hash = getattr(self, "_last_image_hash", self._compute_hash(image_path))
        user_id = getattr(self.repository, "user_id", "default")

        if "user_id" in params:
            return save_receipt(
                user_id=user_id,
                image_path=image_path,
                receipt_data=receipt_data,
                image_hash=image_hash,
            )

        receipt_model = receipt_data if isinstance(receipt_data, Receipt) else Receipt.model_validate(receipt_data)
        parsed_response = APIResponse.success(receipt_model)
        token_usage = getattr(getattr(self, "token_usage", None), "input_tokens", 0)
        output_tokens = getattr(getattr(self, "token_usage", None), "output_tokens", 0)
        return save_receipt(
            image_path=image_path,
            ocr_text=getattr(self, "_last_ocr_text", "") or "",
            parsed_response=parsed_response,
            input_tokens=token_usage,
            output_tokens=output_tokens,
        )

# -----------------------------
# Backwards compatibility API
# -----------------------------

def process_receipt(image_path: str, ocr_service, parser, validator, repository):
    processor = Processor(ocr_service, parser, validator, repository)
    return processor.process(image_path)


def process_single_image(image_path: str, ocr_service, parser, validator, repository):
    return process_receipt(image_path, ocr_service, parser, validator, repository)


def validate_and_get_image_files(paths):
    import os
    from pathlib import Path

    # Handle single path (string or Path object)
    if isinstance(paths, (str, Path)):
        paths = [paths]

    valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    # Collect all image files
    image_files = []
    for p in paths:
        # Convert to Path object if string
        path_obj = Path(p) if isinstance(p, str) else p

        if not path_obj.exists():
            continue

        # If it's a directory, scan for images
        if path_obj.is_dir():
            for ext in valid_extensions:
                image_files.extend(path_obj.glob(f"*{ext}"))
                image_files.extend(path_obj.glob(f"*{ext.upper()}"))
        # If it's a file with valid extension, add it
        elif path_obj.is_file() and path_obj.suffix.lower() in valid_extensions:
            image_files.append(path_obj)

    # Return list of strings for compatibility
    return [str(p) for p in image_files]


# -----------------------------
# Printing / Reporting Helpers
# -----------------------------

def print_processing_result(result):
    """Minimal safe output for a single result"""
    print(f"Status: {result.status}")
    print(f"Receipt ID: {result.receipt_id}")
    print(f"Retry Count: {result.retry_count}")


def print_batch_summary(results):
    """Summarize batch results"""
    total = len(results)
    success = sum(1 for r in results if r.status == PipelineStatus.SUCCESS)
    partial = sum(1 for r in results if r.status == PipelineStatus.PARTIAL)
    failure = sum(1 for r in results if r.status == PipelineStatus.FAILURE)

    print(f"Total: {total}")
    print(f"Success: {success}")
    print(f"Partial: {partial}")
    print(f"Failure: {failure}")


def print_token_usage_summary(*args, **kwargs):
    """Stub for compatibility"""
    print("Token usage summary not implemented (stub)")


def save_token_usage_to_persistence(*args, **kwargs):
    """Stub for compatibility"""
    # no-op
    return None


def print_usage_summary(*args, **kwargs):
    """Stub for compatibility"""
    print("Usage summary not implemented (stub)")


# Alias for legacy imports
ReceiptProcessor = Processor
