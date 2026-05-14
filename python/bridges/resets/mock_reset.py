from __future__ import annotations

from bridges.core.reset_strategy import ResetInfo


class MockReset:
    """Mock ResetStrategy for testing. Instant, no-op."""

    def __init__(self):
        self._reset_count = 0
        self._checkpoint: str | None = None

    def info(self) -> ResetInfo:
        return ResetInfo(
            name="mock",
            is_instant=True,
            typical_duration_sec=0.0,
            supports_checkpoints=False,
        )

    def reset(self) -> None:
        self._reset_count += 1

    def set_checkpoint(self, checkpoint_id: str) -> None:
        self._checkpoint = checkpoint_id

    @property
    def reset_count(self) -> int:
        return self._reset_count
