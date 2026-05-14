from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

from bridges.core.action_sink import ActionSinkInfo


class MockActionSink:
    """Mock ActionSink that records all sent actions in a ring buffer.

    Follows the same history pattern as MockGamepadBackend for consistency.
    """

    def __init__(
        self,
        action_space: gym.Space | None = None,
        max_history: int = 1000,
    ):
        self._action_space = action_space or gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self._history: deque[np.ndarray] = deque(maxlen=max_history)
        self._connected = False

    def info(self) -> ActionSinkInfo:
        return ActionSinkInfo(
            name="mock",
            action_space=self._action_space,
            supports_continuous=True,
            supports_discrete=True,
            platform="any",
        )

    def connect(self) -> None:
        self._connected = True

    def send(self, action: np.ndarray) -> None:
        if not self._connected:
            raise RuntimeError("MockActionSink not connected")
        self._history.append(np.asarray(action, dtype=np.float32).copy())

    def reset(self) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def get_history(self) -> list[np.ndarray]:
        return list(self._history)
