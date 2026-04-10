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


class FakeRepository(FakeComponent):
    """Fake repository with in-memory storage and configurable behavior.
    
    Simulates:
    - Receipt CRUD operations
    - User management
    - Idempotency checks (by image hash or data hash)
    - Query operations
    - Simulated latency and failures
    
    Example:
        >>> repo = FakeRepository()
        >>> repo.seed_receipt(ReceiptBuilder().with_total(25.50).build())
        >>> repo.find_by_image_hash("abc123")  # Returns seeded receipt
        >>> repo.was_saved_with_total(25.50)   # True
    """
    
    def __init__(self):
        super().__init__()
        self._receipts: Dict[str, SavedReceipt] = {}  # id -> receipt
        self._users: Dict[str, SavedUser] = {}  # id -> user
        self._image_hash_index: Dict[str, str] = {}  # image_hash -> receipt_id
        self._data_hash_index: Dict[str, str] = {}  # data_hash -> receipt_id
        self._user_email_index: Dict[str, str] = {}  # email -> user_id
        
        self._should_fail_on: Dict[str, Optional[Exception]] = {}
        self._latency_ms: float = 0.0
        self._auto_generate_ids: bool = True
    
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
                'merchant': getattr(receipt_dto, 'merchant_name', ''),
                'total': getattr(receipt_dto, 'total_amount', 0),
                'date': getattr(receipt_dto, 'receipt_date', ''),
                'items': [],
            },
            receipt_data_hash=data_hash,
            status=getattr(receipt_dto, 'status', 'success'),
            created_at=getattr(receipt_dto, 'created_at', datetime.now()),
            updated_at=getattr(receipt_dto, 'updated_at', datetime.now()),
        )
        
        self._receipts[receipt_id] = saved
        self._image_hash_index[image_hash] = receipt_id
        self._data_hash_index[data_hash] = receipt_id
        
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
    
    # Receipt operations
    
    def save_receipt(
        self,
        user_id: str,
        image_path: str,
        receipt_data: Dict[str, Any],
        image_hash: Optional[str] = None
    ) -> Any:
        """Save a receipt to the repository.
        
        Args:
            user_id: User ID to associate with receipt
            image_path: Path to the source image
            receipt_data: Parsed receipt data
            image_hash: Optional pre-computed image hash
            
        Returns:
            ReceiptDTO-like object
        """
        start = time.time()
        self._simulate_latency()
        self._check_should_fail("save")
        
        receipt_id = str(uuid4()) if self._auto_generate_ids else "manual_id"
        data_hash = receipt_data.get('data_hash') or f"data_{receipt_id}"
        img_hash = image_hash or f"img_{receipt_id}"
        
        # Check idempotency - data hash
        existing_id = self._data_hash_index.get(data_hash)
        if existing_id and existing_id in self._receipts:
            existing = self._receipts[existing_id]
            result = self._to_dto(existing)
            duration_ms = (time.time() - start) * 1000
            self._record_call(
                "save_receipt",
                (user_id, image_path, receipt_data),
                {"image_hash": image_hash},
                result=result,
                duration_ms=duration_ms
            )
            return result
        
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
        
        self._receipts[receipt_id] = saved
        self._image_hash_index[img_hash] = receipt_id
        self._data_hash_index[data_hash] = receipt_id
        
        result = self._to_dto(saved)
        duration_ms = (time.time() - start) * 1000
        
        self._record_call(
            "save_receipt",
            (user_id, image_path, receipt_data),
            {"image_hash": image_hash},
            result=result,
            duration_ms=duration_ms
        )
        
        return result
    
    def find_by_image_hash(self, image_hash: str) -> Optional[Any]:
        """Find receipt by image hash (idempotency check)."""
        start = time.time()
        self._simulate_latency()
        self._check_should_fail("find")
        
        receipt_id = self._image_hash_index.get(image_hash)
        result = None
        
        if receipt_id and receipt_id in self._receipts:
            result = self._to_dto(self._receipts[receipt_id])
        
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
        
        receipt_id = self._data_hash_index.get(data_hash)
        result = None
        
        if receipt_id and receipt_id in self._receipts:
            result = self._to_dto(self._receipts[receipt_id])
        
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
        
        saved = self._receipts.get(receipt_id)
        result = self._to_dto(saved) if saved else None
        
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
        
        saved = self._receipts.get(receipt_id)
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
                self.merchant_name = saved_receipt.receipt_data.get('merchant', '')
                self.total_amount = saved_receipt.receipt_data.get('total')
                self.receipt_date = saved_receipt.receipt_data.get('date')
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
        return list(self._receipts.values())
    
    def was_saved_with_total(self, total: float) -> bool:
        """Check if any receipt was saved with given total."""
        return any(
            r.receipt_data.get('total') == total
            for r in self._receipts.values()
        )
    
    def was_saved_with_merchant(self, merchant: str) -> bool:
        """Check if any receipt was saved with given merchant."""
        return any(
            r.receipt_data.get('merchant') == merchant
            for r in self._receipts.values()
        )
    
    def get_save_count(self) -> int:
        """Get number of save operations performed."""
        return len([c for c in self._calls if c.method == "save_receipt"])
    
    def idempotency_was_checked(self) -> bool:
        """Check if idempotency lookup was performed."""
        return any(
            c.method in ("find_by_image_hash", "find_by_data_hash")
            for c in self._calls
        )
    
    def duplicate_was_detected(self) -> bool:
        """Check if a duplicate receipt was detected."""
        # Check if any save call returned an existing receipt
        for call in self.get_calls("save_receipt"):
            if call.result and hasattr(call.result, 'id'):
                # If id existed before this call, it was a duplicate
                call_time = call.timestamp
                earlier_saves = [
                    c for c in self._calls
                    if c.method == "save_receipt"
                    and c.timestamp < call_time
                    and c.result
                    and getattr(c.result, 'id', None) == getattr(call.result, 'id', None)
                ]
                if earlier_saves:
                    return True
        return False
