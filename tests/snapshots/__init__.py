"""Snapshot testing for parsing verification.

Compares parsed receipt structures against stored fixtures.
Focuses on structured correctness, not brittle string comparisons.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass


@dataclass
class SnapshotComparison:
    """Result of snapshot comparison."""
    matches: bool
    differences: List[str]
    actual: Dict[str, Any]
    expected: Dict[str, Any]


class SnapshotHelper:
    """Helper for snapshot testing of parsed receipts.
    
    Usage:
        helper = SnapshotHelper()
        result = helper.compare(parsed_receipt, "simple_receipt")
        assert result.matches, result.differences
    """
    
    def __init__(self, snapshot_dir: Optional[Path] = None):
        """Initialize with snapshot directory."""
        if snapshot_dir is None:
            snapshot_dir = Path(__file__).parent
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(exist_ok=True)
    
    def compare(
        self,
        actual: Dict[str, Any],
        snapshot_name: str,
        ignore_fields: Optional[List[str]] = None,
        ignore_order_fields: Optional[List[str]] = None
    ) -> SnapshotComparison:
        """Compare actual result against stored snapshot.
        
        Args:
            actual: Parsed receipt data
            snapshot_name: Name of snapshot file (without .json)
            ignore_fields: Fields to ignore in comparison
            ignore_order_fields: List fields where order doesn't matter
            
        Returns:
            SnapshotComparison with match status and differences
        """
        snapshot_path = self.snapshot_dir / f"{snapshot_name}.json"
        
        if not snapshot_path.exists():
            # Create snapshot if doesn't exist
            self._save_snapshot(actual, snapshot_path)
            return SnapshotComparison(
                matches=True,
                differences=[],
                actual=actual,
                expected=actual
            )
        
        expected = self._load_snapshot(snapshot_path)
        differences = self._find_differences(
            actual, expected,
            ignore_fields=ignore_fields or [],
            ignore_order_fields=ignore_order_fields or []
        )
        
        return SnapshotComparison(
            matches=len(differences) == 0,
            differences=differences,
            actual=actual,
            expected=expected
        )
    
    def update_snapshot(self, actual: Dict[str, Any], snapshot_name: str) -> None:
        """Update stored snapshot with new data."""
        snapshot_path = self.snapshot_dir / f"{snapshot_name}.json"
        self._save_snapshot(actual, snapshot_path)
    
    def _load_snapshot(self, path: Path) -> Dict[str, Any]:
        """Load snapshot from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_snapshot(self, data: Dict[str, Any], path: Path) -> None:
        """Save snapshot to JSON file with formatting."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
    
    def _find_differences(
        self,
        actual: Any,
        expected: Any,
        path: str = "",
        ignore_fields: List[str] = None,
        ignore_order_fields: List[str] = None
    ) -> List[str]:
        """Recursively find differences between structures."""
        differences = []
        ignore_fields = ignore_fields or []
        ignore_order_fields = ignore_order_fields or []
        
        # Skip ignored fields
        current_field = path.split('.')[-1] if '.' in path else path
        if current_field in ignore_fields:
            return differences
        
        # Type mismatch
        if type(actual) != type(expected):
            differences.append(f"{path}: type mismatch {type(actual).__name__} vs {type(expected).__name__}")
            return differences
        
        # Dict comparison
        if isinstance(actual, dict):
            all_keys = set(actual.keys()) | set(expected.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key in ignore_fields:
                    continue
                if key not in actual:
                    differences.append(f"{new_path}: missing in actual")
                elif key not in expected:
                    differences.append(f"{new_path}: unexpected in actual")
                else:
                    differences.extend(
                        self._find_differences(
                            actual[key], expected[key], new_path,
                            ignore_fields, ignore_order_fields
                        )
                    )
        
        # List comparison
        elif isinstance(actual, list):
            if len(actual) != len(expected):
                differences.append(f"{path}: length {len(actual)} vs {len(expected)}")
            else:
                # Check if we should ignore order
                should_ignore_order = current_field in ignore_order_fields
                
                if should_ignore_order:
                    # Sort both lists by their string representation for comparison
                    actual_sorted = sorted(actual, key=lambda x: json.dumps(x, sort_keys=True))
                    expected_sorted = sorted(expected, key=lambda x: json.dumps(x, sort_keys=True))
                    for i, (a, e) in enumerate(zip(actual_sorted, expected_sorted)):
                        new_path = f"{path}[{i}]"
                        differences.extend(
                            self._find_differences(a, e, new_path, ignore_fields, ignore_order_fields)
                        )
                else:
                    for i, (a, e) in enumerate(zip(actual, expected)):
                        new_path = f"{path}[{i}]"
                        differences.extend(
                            self._find_differences(a, e, new_path, ignore_fields, ignore_order_fields)
                        )
        
        # Value comparison
        elif actual != expected:
            differences.append(f"{path}: {repr(actual)} != {repr(expected)}")
        
        return differences
    
    def assert_matches_snapshot(
        self,
        actual: Dict[str, Any],
        snapshot_name: str,
        ignore_fields: Optional[List[str]] = None,
        ignore_order_fields: Optional[List[str]] = None
    ) -> None:
        """Assert that actual matches snapshot, raising detailed error on mismatch."""
        result = self.compare(actual, snapshot_name, ignore_fields, ignore_order_fields)
        
        if not result.matches:
            msg = f"Snapshot '{snapshot_name}' mismatch:\n"
            msg += "\n".join(f"  - {d}" for d in result.differences)
            msg += f"\n\nTo update snapshot: helper.update_snapshot(actual, '{snapshot_name}')"
            raise AssertionError(msg)


def normalize_receipt_for_snapshot(receipt_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize receipt data for consistent snapshot comparison.
    
    - Sorts items by name
    - Normalizes numeric values
    - Ensures consistent date format
    """
    normalized = {}
    
    for key, value in receipt_data.items():
        if key == 'items' and isinstance(value, list):
            # Sort items by name for consistency
            normalized[key] = sorted(
                [normalize_receipt_for_snapshot(item) for item in value],
                key=lambda x: x.get('name', '') or x.get('description', '')
            )
        elif isinstance(value, float):
            # Round floats for consistency
            normalized[key] = round(value, 2)
        elif isinstance(value, dict):
            normalized[key] = normalize_receipt_for_snapshot(value)
        elif isinstance(value, list):
            normalized[key] = [normalize_receipt_for_snapshot(item) for item in value]
        else:
            normalized[key] = value
    
    return normalized
