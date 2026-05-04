from typing import Any, Optional

from pipeline.duplicate_detector import DuplicateDetector
from pipeline.persistence_gateway import ReceiptPersistenceGateway
from pipeline.receipt_data_mapper import ReceiptDataMapper
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
        self.receipt_data_mapper = ReceiptDataMapper()
        self.duplicate_detector = DuplicateDetector(repository)
        self.persistence_gateway = ReceiptPersistenceGateway(repository)

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
        # Update repository user_id where possible (try attribute, then method)
        try:
            setattr(self.repository, 'user_id', email)
        except Exception:
            try:
                method = getattr(self.repository, 'set_user_id', None)
                if callable(method):
                    method(email)
            except Exception:
                # Repository does not support updating user id; ignore
                pass

    def get_user_context(self) -> dict:
        """Get current user context."""
        # Prefer repository-provided user_id when available
        try:
            repo_user = getattr(self.repository, 'user_id')
            if repo_user:
                return {"email": repo_user, "user_id": repo_user}
        except Exception:
            pass
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
            if isinstance(existing, dict):
                result.receipt_id = existing.get("id")
                result.receipt_data = existing
            else:
                result.receipt_id = getattr(existing, "id", None)
                # Try to convert to dict when possible
                result.receipt_data = getattr(existing, "__dict__", None) or existing
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
            # Copy token usage from ParsingResult to processor (ParsingResult ensures token_usage exists)
            tu = getattr(parsed, 'token_usage', None)
            if tu:
                self.token_usage = tu

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
        # Prefer explicit validator interface methods in order of specificity
        validate_fn = getattr(self.validator, 'validate_receipt', None) or getattr(self.validator, 'validate', None)
        if callable(validate_fn):
            validation = validate_fn(parsed)
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
        # Normalize receipt_data shapes (dict, object with attributes, or DTO)
        parsed_receipt = {}
        if isinstance(receipt_data, dict) and 'parsed' in receipt_data:
            parsed_receipt = dict(receipt_data['parsed'])
            tu = receipt_data.get('token_usage')
            if tu:
                parsed_receipt['_token_usage'] = {
                    'input_tokens': tu.input_tokens,
                    'output_tokens': tu.output_tokens,
                    'total_tokens': tu.input_tokens + tu.output_tokens,
                }
        else:
            # Try object-style access
            parsed_obj = getattr(receipt_data, 'parsed', None)
            if parsed_obj is not None:
                parsed_receipt = dict(parsed_obj) if isinstance(parsed_obj, dict) else getattr(parsed_obj, '__dict__', {})
                tu = getattr(receipt_data, 'token_usage', None)
                if tu:
                    parsed_receipt['_token_usage'] = {
                        'input_tokens': tu.input_tokens,
                        'output_tokens': tu.output_tokens,
                        'total_tokens': tu.input_tokens + tu.output_tokens,
                    }
            else:
                rd = getattr(receipt_data, 'receipt_data', None)
                if rd is not None:
                    parsed_receipt = rd if isinstance(rd, dict) else getattr(rd, '__dict__', {})
                else:
                    parsed_receipt = receipt_data if isinstance(receipt_data, dict) else getattr(receipt_data, '__dict__', {})

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

            saved = self.retry_service.execute_with_retry(
                self._save_receipt,
                receipt_data,
                error_types=(Exception,),
            )

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
        return self.persistence_gateway.extract_id(saved)

    def _record_trace(self, stage: str, success: bool, data: Optional[dict] = None):
        self.trace.append(TraceEntry(stage, success, data))

    def _compute_hash(self, image_path: str) -> str:
        return self.duplicate_detector.compute_hash(image_path)

    def _find_existing_receipt(self, image_path: str, image_hash: str) -> Optional[Any]:
        return self.duplicate_detector.find_existing(image_path, image_hash)

    def _extract_receipt_data(self, parsed: Any) -> Any:
        return self.receipt_data_mapper.extract(parsed)

    def _to_plain_data(self, value: Any) -> Any:
        return self.receipt_data_mapper.to_plain_data(value)

    def _save_receipt(self, receipt_data: Any) -> Any:
        image_path = getattr(self, "_last_image_path", "unknown")
        image_hash = getattr(self, "_last_image_hash", self._compute_hash(image_path))
        return self.persistence_gateway.save(
            receipt_data=receipt_data,
            image_path=image_path,
            image_hash=image_hash,
            ocr_text=getattr(self, "_last_ocr_text", "") or "",
            token_usage=self.token_usage,
        )

# No backwards compatibility stubs - use the main Processor class directly

# Alias for legacy imports
ReceiptProcessor = Processor
