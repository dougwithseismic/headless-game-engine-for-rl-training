from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ResetInfo:
    name: str
    is_instant: bool
    typical_duration_sec: float
    supports_checkpoints: bool


@runtime_checkable
class ResetStrategy(Protocol):
    """Interface for restarting game episodes.

    Implementations handle the full reset sequence -- save state loads,
    keyboard macros, API calls, etc. reset() blocks until the game is
    ready for a new episode.
    """

    def info(self) -> ResetInfo: ...

    def reset(self) -> None:
        """Execute a reset. Blocks until the game is ready."""
        ...

    def set_checkpoint(self, checkpoint_id: str) -> None:
        """Set which checkpoint/save state to reset to.
        No-op for strategies that don't support checkpoints.
        """
        ...
