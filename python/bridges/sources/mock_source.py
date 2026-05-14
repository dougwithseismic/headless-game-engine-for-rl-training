from __future__ import annotations

import gymnasium as gym
import numpy as np

from bridges.core.obs_source import ObservationSourceInfo, FeatureGroup


class MockObservationSource:
    """Mock ObservationSource that returns random or scripted observations."""

    def __init__(
        self,
        observation_space: gym.Space | None = None,
        terminal_after: int | None = None,
        feature_names: list[str] | None = None,
        feature_groups: list[FeatureGroup] | None = None,
    ):
        self._obs_space = observation_space or gym.spaces.Box(
            -1, 1, shape=(10,), dtype=np.float32
        )
        self._terminal_after = terminal_after
        self._feature_names = feature_names
        self._feature_groups = feature_groups
        self._step_count = 0
        self._connected = False

    def info(self) -> ObservationSourceInfo:
        return ObservationSourceInfo(
            name="mock",
            observation_space=self._obs_space,
            native_hz=None,
            platform="any",
            feature_names=self._feature_names,
            feature_groups=self._feature_groups,
        )

    def connect(self) -> None:
        self._connected = True
        self._step_count = 0

    def read(self) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("MockObservationSource not connected")
        self._step_count += 1
        return self._obs_space.sample()

    def is_terminal(self) -> bool:
        if self._terminal_after is not None:
            return self._step_count >= self._terminal_after
        return False

    def disconnect(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def step_count(self) -> int:
        return self._step_count
