from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import gymnasium as gym
import numpy as np


@dataclass
class FeatureGroup:
    """A named slice of the flat observation vector."""
    name: str
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)


@dataclass
class ObservationSourceInfo:
    name: str
    observation_space: gym.Space
    native_hz: float | None  # None = as-fast-as-possible
    platform: str
    feature_names: list[str] | None = None
    feature_groups: list[FeatureGroup] | None = None


@runtime_checkable
class ObservationSource(Protocol):
    """Interface for reading game state from an external game.

    Implementations handle normalization internally -- read() returns
    arrays ready for the policy network.
    """

    def info(self) -> ObservationSourceInfo: ...

    def connect(self) -> None: ...

    def read(self) -> np.ndarray:
        """Read the current observation. Blocks until fresh data is available."""
        ...

    def is_terminal(self) -> bool:
        """Check if the game is in a terminal state."""
        ...

    def disconnect(self) -> None: ...
