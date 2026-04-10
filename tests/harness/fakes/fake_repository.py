"""Fake repository for deterministic persistence testing."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from uuid import uuid4
import time

from .fake_component import FakeComponent, ConfigurationError


@dataclass
class SavedReceipt:
    """Internal storage format for fake repository."""
    id: str
    user_id: str
    image_path: str
    image_hash: str
    receipt_data: Dict[str, Any]
    receipt_data_hash: str
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass
class SavedUser:
    """Internal storage format for fake users."""
    id: str
    email: str
    created_at: datetime
    is_active: bool = True


@dataclass
class RepositoryMetrics:
    """Metrics for repository operations."""
    save_attempts: int = 0  # Number of save calls made
    actual_writes: int = 0   # Number of actual DB writes (non-duplicates)
    duplicate_detections: int = 0  # Number of duplicate detection events
    constraint_violations: int = 0  # Number of constraint violations


class FakeRepository(FakeComponent):
    """Enhanced fake repository with content-hash deduplication and metrics.

    Simulates:
    - Receipt CRUD operations with content-hash deduplication
    - User management
    - Idempotency checks (by image hash or data hash)
    - Query operations
    - Simulated latency, failures, and constraint violations
    - Detailed operation metrics

    Example:
        >>> repo = FakeRepository()
        >>> repo.seed_receipt(ReceiptBuilder().with_total(25.50).build())
        >>> repo.find_by_image_hash("abc123")  # Returns seeded receipt
        >>> repo.was_saved_with_total(25.50)   # True
        >>> repo.get_metrics().save_attempts   # 1
        >>> repo.get_metrics().actual_writes   # 1
    """

    def __init__(self):
        super().__init__()
        # Primary storage: content_hash -> receipt (enforces deduplication)
        self._content_store: Dict[str, SavedReceipt] = {}  # data_hash -> receipt
        # Secondary indexes for lookups
        self._id_index: Dict[str, str] = {}  # receipt_id -> data_hash
        self._image_hash_index: Dict[str, str] = {}  # image_hash -> data_hash
        self._user_receipts: Dict[str, List[str]] = {}  # user_id -> [data_hash, ...]

        # User storage
        self._users: Dict[str, SavedUser] = {}  # id -> user
        self._user_email_index: Dict[str, str] = {}  # email -> user_id

        # Configuration
        self._should_fail_on: Dict[str, Optional[Exception]] = {}
        self._constraint_violations: Dict[str, Exception] = {}
        self._latency_ms: float = 0.0
        self._auto_generate_ids: bool = True

        # Metrics tracking
        self._metrics = RepositoryMetrics()

    def seed_receipt(self, receipt_dto: Any) -> "FakeRepository":
        """Pre-populate repository with a receipt.

        Args:
            receipt_dto: ReceiptDTO or dict with receipt data

        Returns:
            Self for chaining
        """
        # Convert DTO to internal format
        receipt_id = getattr(receipt_dto, 'id', None) or str(uuid4())
        user_id = getattr(receipt_dto, 'user_id', 'default_user')
        image_hash = getattr(receipt_dto, 'image_hash', f"hash_{receipt_id}")
        data_hash = getattr(receipt_dto, 'receipt_data_hash', f"data_{receipt_id}")

        saved = SavedReceipt(
            id=receipt_id,
            user_id=user_id,
            image_path=getattr(receipt_dto, 'image_path', 'test.jpg'),
            image_hash=image_hash,
            receipt_data={
                'merchant_name': getattr(receipt_dto, 'merchant_name', ''),
                'total_amount': getattr(receipt_dto, 'total_amount', 0),
                'receipt_date': getattr(receipt_dto, 'receipt_date', ''),
                'items': getattr(receipt_dto, 'items', []),
            },
            receipt_data_hash=data_hash,
            status=getattr(receipt_dto, 'status', 'success'),
            created_at=getattr(receipt_dto, 'created_at', datetime.now()),
            updated_at=getattr(receipt_dto, 'updated_at', datetime.now()),
        )

        # Store by content hash (enforces deduplication)
        self._content_store[data_hash] = saved
        self._id_index[receipt_id] = data_hash
        self._image_hash_index[image_hash] = data_hash

        # Track per-user receipts
        if user_id not in self._user_receipts:
            self._user_receipts[user_id] = []
        if data_hash not in self._user_receipts[user_id]:
            self._user_receipts[user_id].append(data_hash)

        return self

    def seed_user(self, user_dto: Any) -> "FakeRepository":
        """Pre-populate repository with a user."""
        user_id = getattr(user_dto, 'id', None) or str(uuid4())
        email = getattr(user_dto, 'email', f"user_{user_id}@test.com")

        saved = SavedUser(
            id=user_id,
            email=email,
            created_at=getattr(user_dto, 'created_at', datetime.now()),
            is_active=getattr(user_dto, 'is_active', True),
        )

        self._users[user_id] = saved
        self._user_email_index[email] = user_id

        return self

    def set_should_fail_on(
        self,
        operation: str,
        exception: Optional[Exception] = None
    ) -> "FakeRepository":
        """Configure an operation to fail.

        Args:
            operation: Operation name (save, update, find, etc.)
            exception: Exception to raise, or None to clear
        """
        self._should_fail_on[operation] = exception
        return self

    def set_constraint_violation(
        self,
        constraint_name: str,
        exception: Exception
    ) -> "FakeRepository":
        """Configure a constraint violation to simulate.

        Args:
            constraint_name: Name of constraint (unique_image_hash, unique_data_hash, etc.)
            exception: Exception to raise when constraint is violated
        """
        self._constraint_violations[constraint_name] = exception
        return self

    def set_latency_ms(self, ms: float) -> "FakeRepository":
        """Simulate database latency."""
        self._latency_ms = ms
        return self

    def _simulate_latency(self) -> None:
        """Apply configured latency."""
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)

    def _check_should_fail(self, operation: str) -> None:
        """Check if operation should fail and raise if so."""
        exc = self._should_fail_on.get(operation)
        if exc:
            raise exc

    def _check_constraints(self, image_hash: str, data_hash: str) -> None:
        """Check if any constraints would be violated."""
        # Check unique image_hash constraint
        if "unique_image_hash" in self._constraint_violations:
            if image_hash in self._image_hash_index:
                self._metrics.constraint_violations += 1
                raise self._constraint_violations["unique_image_hash"]

        # Check unique data_hash constraint
        if "unique_data_hash" in self._constraint_violations:
            if data_hash in self._content_store:
                self._metrics.constraint_violations += 1
                raise self._constraint_violations["unique_data_hash"]

    # New API methods

    def get_all(self) -> List[Any]:
        """Get all receipts as DTOs."""
        return [self._to_dto(r) for r in self._content_store.values()]

    def get_by_hash(self, data_hash: str) -> Optional[Any]:
        """Get receipt by content/data hash."""
        receipt = self._content_store.get(data_hash)
        return self._to_dto(receipt) if receipt else None

    def get_by_image_hash(self, image_hash: str) -> Optional[Any]:
        """Get receipt by image hash."""
        data_hash = self._image_hash_index.get(image_hash)
        if data_hash:
            return self.get_by_hash(data_hash)
        return None

    def clear(self) -> "FakeRepository":
        """Clear all data and reset metrics."""
        self._content_store.clear()
        self._id_index.clear()
        self._image_hash_index.clear()
        self._user_receipts.clear()
        self._users.clear()
        self._user_email_index.clear()
        self._metrics = RepositoryMetrics()
        return self

    def get_metrics(self) -> RepositoryMetrics:
        """Get current repository metrics."""
        return self._metrics

    # Receipt operations

    def save_receipt(
        self,
        user_id: str,
        image_path: str,
        receipt_data: Dict[str, Any],
        image_hash: Optional[str] = None
    ) -> Any:
        """Save a receipt to the repository with content-hash deduplication.

        Args:
            user_id: User ID to associate with receipt
            image_path: Path to the source image
            receipt_data: Parsed receipt data
            image_hash: Optional pre-computed image hash

        Returns:
            ReceiptDTO-like object (existing if duplicate, new if unique)
        """
        start = time.time()
        self._simulate_latency()
        self._check_should_fail("save")

        # Track save attempt
        self._metrics.save_attempts += 1

        # Generate IDs and hashes
        receipt_id = str(uuid4()) if self._auto_generate_ids else "manual_id"
        data_hash = receipt_data.get('data_hash') or self._compute_data_hash(receipt_data)
        img_hash = image_hash or f"img_{receipt_id}"

        # Check constraint violations (if configured)
        self._check_constraints(img_hash, data_hash)

        # Check idempotency by content hash
        existing = self._content_store.get(data_hash)
        if existing:
            # Duplicate detected - return existing without writing
            self._metrics.duplicate_detections += 1
            result = self._to_dto(existing)
            duration_ms = (time.time() - start) * 1000
            self._record_call(
                "save_receipt",
                (user_id, image_path, receipt_data),
                {"image_hash": image_hash, "duplicate": True},
                result=result,
                duration_ms=duration_ms
            )
            return result

        # New receipt - perform actual write
        self._metrics.actual_writes += 1

        saved = SavedReceipt(
            id=receipt_id,
            user_id=user_id,
            image_path=image_path,
            image_hash=img_hash,
            receipt_data=receipt_data,
            receipt_data_hash=data_hash,
            status='success',
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Store by content hash (enforces deduplication)
        self._content_store[data_hash] = saved
        self._id_index[receipt_id] = data_hash
        self._image_hash_index[img_hash] = data_hash

        # Track per-user receipts
        if user_id not in self._user_receipts:
            self._user_receipts[user_id] = []
        self._user_receipts[user_id].append(data_hash)

        result = self._to_dto(saved)
        duration_ms = (time.time() - start) * 1000

        self._record_call(
            "save_receipt",
            (user_id, image_path, receipt_data),
            {"image_hash": image_hash, "duplicate": False},
            result=result,
            duration_ms=duration_ms
        )

        return result

    def _compute_data_hash(self, receipt_data: Dict[str, Any]) -> str:
        """Compute a deterministic hash from receipt data."""
        import hashlib
        import json

        # Create deterministic string from key fields
        key_fields = {
            'merchant_name': receipt_data.get('merchant_name', ''),
            'total_amount': receipt_data.get('total_amount', 0),
            'receipt_date': receipt_data.get('receipt_date', ''),
            'items': receipt_data.get('items', []),
        }
        data_str = json.dumps(key_fields, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def find_by_image_hash(self, image_hash: str) -> Optional[Any]:
        """Find receipt by image hash (idempotency check)."""
        start = time.time()
        self._simulate_latency()
        self._check_should_fail("find")

        # Use new content-hash based lookup
        data_hash = self._image_hash_index.get(image_hash)
        result = None

        if data_hash:
            receipt = self._content_store.get(data_hash)
            if receipt:
                result = self._to_dto(receipt)

        duration_ms = (time.time() - start) * 1000
        self._record_call(
            "find_by_image_hash",
            (image_hash,),
            {},
            result=result,
            duration_ms=duration_ms
        )

        return result

    def find_by_data_hash(self, data_hash: str) -> Optional[Any]:
        """Find receipt by data hash (idempotency check)."""
        start = time.time()
        self._simulate_latency()
        self._check_should_fail("find")

        receipt = self._content_store.get(data_hash)
        result = self._to_dto(receipt) if receipt else None

        duration_ms = (time.time() - start) * 1000
        self._record_call(
            "find_by_data_hash",
            (data_hash,),
            {},
            result=result,
            duration_ms=duration_ms
        )

        return result

    def get_receipt_by_id(self, receipt_id: str) -> Optional[Any]:
        """Get receipt by ID."""
        start = time.time()
        self._simulate_latency()

        # Look up by ID index then get from content store
        data_hash = self._id_index.get(receipt_id)
        result = None

        if data_hash:
            saved = self._content_store.get(data_hash)
            if saved:
                result = self._to_dto(saved)

        duration_ms = (time.time() - start) * 1000
        self._record_call(
            "get_receipt_by_id",
            (receipt_id,),
            {},
            result=result,
            duration_ms=duration_ms
        )

        return result

    def update_receipt(
        self,
        receipt_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Any]:
        """Update a receipt."""
        start = time.time()
        self._simulate_latency()
        self._check_should_fail("update")

        # Find receipt by ID
        data_hash = self._id_index.get(receipt_id)
        if not data_hash:
            duration_ms = (time.time() - start) * 1000
            self._record_call(
                "update_receipt",
                (receipt_id, updates),
                {},
                result=None,
                duration_ms=duration_ms
            )
            return None

        saved = self._content_store.get(data_hash)
        if not saved:
            duration_ms = (time.time() - start) * 1000
            self._record_call(
                "update_receipt",
                (receipt_id, updates),
                {},
                result=None,
                duration_ms=duration_ms
            )
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(saved, key):
                setattr(saved, key, value)
            elif key in saved.receipt_data:
                saved.receipt_data[key] = value

        saved.updated_at = datetime.now()

        result = self._to_dto(saved)
        duration_ms = (time.time() - start) * 1000

        self._record_call(
            "update_receipt",
            (receipt_id, updates),
            {},
            result=result,
            duration_ms=duration_ms
        )

        return result

    def _to_dto(self, saved: SavedReceipt) -> Any:
        """Convert internal format to DTO-like object."""
        # Return a simple object that behaves like ReceiptDTO
        class ReceiptDTO:
            def __init__(self, saved_receipt):
                self.id = saved_receipt.id
                self.user_id = saved_receipt.user_id
                self.image_path = saved_receipt.image_path
                self.image_hash = saved_receipt.image_hash
                self.receipt_data_hash = saved_receipt.receipt_data_hash
                self.status = saved_receipt.status
                # receipt_data keys come from parser: merchant_name, total_amount, receipt_date
                self.merchant_name = saved_receipt.receipt_data.get('merchant_name', '')
                self.total_amount = saved_receipt.receipt_data.get('total_amount')
                self.receipt_date = saved_receipt.receipt_data.get('receipt_date')
                self.created_at = saved_receipt.created_at
                self.updated_at = saved_receipt.updated_at

        return ReceiptDTO(saved)

    # User operations

    def get_or_create_user(self, email: str) -> Any:
        """Get existing user or create new one."""
        start = time.time()
        self._simulate_latency()
        self._check_should_fail("user")

        user_id = self._user_email_index.get(email)

        if user_id and user_id in self._users:
            user = self._users[user_id]
        else:
            # Create new user
            user_id = str(uuid4())
            user = SavedUser(
                id=user_id,
                email=email,
                created_at=datetime.now(),
                is_active=True
            )
            self._users[user_id] = user
            self._user_email_index[email] = user_id

        result = self._user_to_dto(user)
        duration_ms = (time.time() - start) * 1000

        self._record_call(
            "get_or_create_user",
            (email,),
            {},
            result=result,
            duration_ms=duration_ms
        )

        return result

    def _user_to_dto(self, saved: SavedUser) -> Any:
        """Convert user to DTO-like object."""
        class UserDTO:
            def __init__(self, saved_user):
                self.id = saved_user.id
                self.email = saved_user.email
                self.created_at = saved_user.created_at
                self.is_active = saved_user.is_active

        return UserDTO(saved)

    # Assertion helpers

    def get_saved_receipts(self) -> List[SavedReceipt]:
        """Get all saved receipts."""
        return list(self._content_store.values())

    def was_saved_with_total(self, total: float) -> bool:
        """Check if any receipt was saved with given total."""
        return any(
            r.receipt_data.get('total_amount') == total
            for r in self._content_store.values()
        )

    def was_saved_with_merchant(self, merchant: str) -> bool:
        """Check if any receipt was saved with given merchant."""
        return any(
            r.receipt_data.get('merchant_name') == merchant
            for r in self._content_store.values()
        )

    def get_save_count(self) -> int:
        """Get number of save operations performed."""
        return len([c for c in self._calls if c.method == "save_receipt"])

    def get_actual_write_count(self) -> int:
        """Get number of actual DB writes (excluding duplicates)."""
        return self._metrics.actual_writes

    def get_duplicate_count(self) -> int:
        """Get number of duplicate detections."""
        return self._metrics.duplicate_detections

    def idempotency_was_checked(self) -> bool:
        """Check if idempotency lookup was performed."""
        return any(
            c.method in ("find_by_image_hash", "find_by_data_hash")
            for c in self._calls
        )

    def duplicate_was_detected(self) -> bool:
        """Check if a duplicate receipt was detected."""
        # Check metrics for duplicate detections
        if self._metrics.duplicate_detections > 0:
            return True

        # Also check call history for duplicates
        for call in self.get_calls("save_receipt"):
            if call.kwargs.get("duplicate"):
                return True
        return False
