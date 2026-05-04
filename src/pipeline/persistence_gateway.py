import inspect
from typing import Any, Optional

from api_response import APIResponse
from domain.models.receipt import Receipt


class ReceiptPersistenceGateway:
    """Adapt processor persistence requests to supported repository APIs."""

    def __init__(self, repository: Any):
        self.repository = repository

    def save(
        self,
        receipt_data: Any,
        image_path: str,
        image_hash: str,
        ocr_text: str,
        token_usage: Any = None,
    ) -> Any:
        save_receipt = getattr(self.repository, 'save_receipt', None)
        if callable(save_receipt):
            return self._save_receipt(receipt_data, image_path, image_hash, ocr_text, token_usage)

        save_fn = getattr(self.repository, 'save', None)
        if callable(save_fn):
            return save_fn(receipt_data)

        raise AttributeError("Repository has no save method")

    def extract_id(self, saved: Any) -> Optional[str]:
        if saved is None:
            return None
        if isinstance(saved, tuple) and saved:
            return self.extract_id(saved[0])
        if isinstance(saved, dict):
            return saved.get("id")
        return getattr(saved, "id", None)

    def _save_receipt(
        self,
        receipt_data: Any,
        image_path: str,
        image_hash: str,
        ocr_text: str,
        token_usage: Any = None,
    ) -> Any:
        save_receipt = self.repository.save_receipt
        params = inspect.signature(save_receipt).parameters
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
        return save_receipt(
            image_path=image_path,
            ocr_text=ocr_text or "",
            parsed_response=parsed_response,
            input_tokens=getattr(token_usage, "input_tokens", 0),
            output_tokens=getattr(token_usage, "output_tokens", 0),
        )
