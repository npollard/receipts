"""Hashing utilities for idempotency and deduplication"""

import hashlib
import json
from typing import Dict, Any


def calculate_file_hash(file_path: str, chunk_size: int = 4096) -> str:
    """Calculate SHA-256 hash of a file for deduplication

    Args:
        file_path: Path to the file to hash
        chunk_size: Size of chunks to read (default: 4096)

    Returns:
        SHA-256 hash as hexadecimal string
    """
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except FileNotFoundError:
        # If file doesn't exist, hash the path as fallback
        return hashlib.sha256(file_path.encode()).hexdigest()


def calculate_image_hash(image_path: str) -> str:
    """Calculate SHA-256 hash of image file for deduplication

    Args:
        image_path: Path to the image file

    Returns:
        SHA-256 hash as hexadecimal string
    """
    return calculate_file_hash(image_path)


def calculate_data_hash(data: Dict[str, Any], normalize: bool = True) -> str:
    """Calculate SHA-256 hash of dictionary data

    Args:
        data: Dictionary data to hash
        normalize: Whether to normalize the data for consistent hashing

    Returns:
        SHA-256 hash as hexadecimal string
    """
    if normalize:
        # Create a normalized representation for consistent hashing
        normalized_data = normalize_data_for_hashing(data)
        data_string = json.dumps(normalized_data, sort_keys=True, separators=(',', ':'))
    else:
        data_string = json.dumps(data, sort_keys=True, separators=(',', ':'))

    return hashlib.sha256(data_string.encode()).hexdigest()


def calculate_receipt_data_hash(receipt_data: Dict[str, Any]) -> str:
    """Calculate SHA-256 hash of receipt data for idempotency

    Args:
        receipt_data: Receipt data dictionary

    Returns:
        SHA-256 hash as hexadecimal string
    """
    # Create a normalized representation of receipt data for hashing
    normalized_data = {
        'date': receipt_data.get('date'),
        'total': receipt_data.get('total'),
        'merchant': receipt_data.get('merchant', ''),
        'items': []
    }

    # Normalize items for consistent hashing
    items = receipt_data.get('items', [])
    if items:
        for item in items:
            if isinstance(item, dict):
                normalized_item = {
                    'description': str(item.get('description', '')).lower().strip(),
                    'price': float(item.get('price', 0)),
                    'quantity': float(item.get('quantity', 1))
                }
                normalized_data['items'].append(normalized_item)

        # Sort items by description and price for consistent ordering
        normalized_data['items'].sort(key=lambda x: (x['description'], x['price']))

    return calculate_data_hash(normalized_data, normalize=False)


def normalize_data_for_hashing(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize data dictionary for consistent hashing

    Args:
        data: Dictionary to normalize

    Returns:
        Normalized dictionary
    """
    if not isinstance(data, dict):
        return {'value': str(data)}

    normalized = {}
    for key, value in sorted(data.items()):
        if isinstance(value, dict):
            normalized[key] = normalize_data_for_hashing(value)
        elif isinstance(value, list):
            # Convert list to tuple for sorting if it contains dictionaries
            try:
                normalized_list = [
                    normalize_data_for_hashing(item) if isinstance(item, dict) else str(item)
                    for item in value
                ]
                # Try to sort if possible, otherwise keep original order
                try:
                    normalized[key] = sorted(normalized_list)
                except TypeError:
                    # Can't sort mixed types, keep original order
                    normalized[key] = normalized_list
            except Exception:
                normalized[key] = str(value)
        elif isinstance(value, tuple):
            normalized[key] = tuple(
                normalize_data_for_hashing(item) if isinstance(item, dict) else str(item)
                for item in value
            )
        elif isinstance(value, (int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = str(value)

    return normalized


def calculate_string_hash(text: str) -> str:
    """Calculate SHA-256 hash of a string

    Args:
        text: String to hash

    Returns:
        SHA-256 hash as hexadecimal string
    """
    return hashlib.sha256(text.encode()).hexdigest()
