from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import gymnasium as gym
import numpy as np


@dataclass
class ActionSinkInfo:
    name: str
    action_space: gym.Space
    supports_continuous: bool
    supports_discrete: bool
    platform: str  # "any", "windows", "linux", etc.


@runtime_checkable
class ActionSink(Protocol):
    """Interface for sending actions to an external game.

    Implementations translate raw numpy arrays into game-specific inputs
    (gamepad axes, keyboard presses, emulator buttons, network commands).
    """

    def info(self) -> ActionSinkInfo: ...

    def connect(self) -> None: ...

    def send(self, action: np.ndarray) -> None:
        """Send an action. Array shape must match info().action_space."""
        ...

    def reset(self) -> None:
        """Reset to neutral state (all zeros, all released)."""
        ...

    def disconnect(self) -> None: ...
