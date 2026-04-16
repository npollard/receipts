"""Reusable invariant assertions for pipeline tests.

Validates pipeline behavior against formal contracts.
"""

from typing import List, Optional
from tests.harness.pipeline_harness import PipelineTestHarness, PipelineStatus


class PipelineInvariants:
    """Invariant checker for pipeline execution.

    Usage:
        invariants = PipelineInvariants(scenario)
        invariants.assert_no_duplicate_persistence()
        invariants.assert_retry_within_limits(max_retries=3)
    """

    def __init__(self, scenario):
        """Initialize with executed scenario.

        Args:
            scenario: Executed Scenario instance with _harness reference
        """
        self._scenario = scenario
        self._harness: Optional[PipelineTestHarness] = scenario._harness
        self._result = getattr(scenario, '_result', None)

        if self._harness is None:
            raise RuntimeError("Scenario must be executed before invariant checks")

    def assert_no_duplicate_persistence(self) -> None:
        """INVARIANT-1: No duplicate persistence across retries.

        Ensures same receipt is not persisted multiple times.
        """
        # Check repository metrics for actual writes
        metrics = self._harness.repository.get_metrics()
        actual_writes = metrics.actual_writes
        save_attempts = metrics.save_attempts

        # In a correct implementation, actual_writes should never exceed 1 per unique receipt
        # Duplicates should be detected by content hash
        assert actual_writes <= save_attempts, (
            f"More actual writes ({actual_writes}) than save attempts ({save_attempts}). "
            "Possible duplicate persistence detected."
        )

        # If multiple save attempts, verify they were deduplicated
        if save_attempts > 1:
            assert metrics.duplicate_detections >= save_attempts - actual_writes, (
                f"Expected duplicate detection for retries. "
                f"Attempts: {save_attempts}, Writes: {actual_writes}, "
                f"Detections: {metrics.duplicate_detections}"
            )

    def assert_retry_within_limits(self, max_retries: int = 3) -> None:
        """INVARIANT-2: Retry count ≤ max_attempts.

        Args:
            max_retries: Maximum allowed retry attempts (default 3)
        """
        # Get total stage attempts across all stages
        ocr_attempts = self._scenario.get_stage_attempts("OCR")
        parsing_attempts = self._scenario.get_stage_attempts("PARSE")
        persist_attempts = self._scenario.get_stage_attempts("PERSIST")

        # Each stage type has its own limit
        # OCR: max 3 (original + 2 retries)
        # Parse: max 2 (original + reparse)
        # Persistence: max 3 (original + 2 retries)

        ocr_retries = max(0, ocr_attempts - 1)  # First attempt is not a retry
        parse_retries = max(0, parsing_attempts - 1)
        persist_retries = max(0, persist_attempts - 1)

        assert ocr_retries <= 3, f"OCR retries ({ocr_retries}) exceeds limit (3)"
        assert parse_retries <= 2, f"Parse retries ({parse_retries}) exceeds limit (2)"
        assert persist_retries <= 3, f"Persistence retries ({persist_retries}) exceeds limit (3)"

    def assert_no_post_terminal_execution(self) -> None:
        """INVARIANT-3: No stages execute after terminal success/failure.

        Pipeline should halt immediately after terminal state reached.
        """
        trace = self._harness.trace.get_all()
        if not trace:
            return

        # Find terminal stage (PERSISTENCE success or terminal failure)
        terminal_index = None
        for i, entry in enumerate(trace):
            if entry.stage == "PERSIST" and entry.data.get("status") == "success":
                terminal_index = i
                break
            # Terminal failure indicators
            if entry.data.get("terminal"):
                terminal_index = i
                break

        if terminal_index is not None:
            # No entries should follow terminal stage
            post_terminal = trace[terminal_index + 1:]
            assert len(post_terminal) == 0, (
                f"Stages executed after terminal state: "
                f"{[e.stage for e in post_terminal]}"
            )

    def assert_persistence_after_validation_only(self) -> None:
        """INVARIANT-4: Persistence only occurs after validation passes.

        No persistence should occur if validation failed.
        """
        trace = self._harness.trace.get_all()

        # Find validation and persistence entries
        validation_passed = False
        persistence_before_validation = False

        for entry in trace:
            if entry.stage == "VALIDATE":
                if entry.data.get("status") == "pass":
                    validation_passed = True
            elif entry.stage == "PERSIST":
                if not validation_passed:
                    persistence_before_validation = True

        assert not persistence_before_validation, (
            "Persistence occurred before validation passed"
        )

    def check_all(self, max_retries: int = 3) -> None:
        """Run all invariant checks.

        Args:
            max_retries: Maximum allowed retry attempts
        """
        self.assert_no_duplicate_persistence()
        self.assert_retry_within_limits(max_retries)
        self.assert_no_post_terminal_execution()
        self.assert_persistence_after_validation_only()


def assert_no_duplicate_persistence(scenario) -> None:
    """Convenience function: INVARIANT-1."""
    PipelineInvariants(scenario).assert_no_duplicate_persistence()


def assert_retry_within_limits(scenario, max_retries: int = 3) -> None:
    """Convenience function: INVARIANT-2."""
    PipelineInvariants(scenario).assert_retry_within_limits(max_retries)


def assert_no_post_terminal_execution(scenario) -> None:
    """Convenience function: INVARIANT-3."""
    PipelineInvariants(scenario).assert_no_post_terminal_execution()


def assert_persistence_after_validation_only(scenario) -> None:
    """Convenience function: INVARIANT-4."""
    PipelineInvariants(scenario).assert_persistence_after_validation_only()


def check_all_invariants(scenario, max_retries: int = 3) -> None:
    """Run all invariant checks on scenario.

    Args:
        scenario: Executed Scenario instance
        max_retries: Maximum allowed retry attempts
    """
    PipelineInvariants(scenario).check_all(max_retries)
