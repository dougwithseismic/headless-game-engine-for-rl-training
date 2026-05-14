from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from bridges.core.action_sink import ActionSink
from bridges.core.obs_source import ObservationSource
from bridges.core.reset_strategy import ResetStrategy
from bridges.core.timing import TimingConfig, TimingPolicy, StepTimer


@dataclass
class GameBridgeConfig:
    name: str
    timing: TimingConfig


class GameBridge:
    """Composes ActionSink + ObservationSource + ResetStrategy into a game interface.

    Each external game is defined by providing three components. The bridge
    handles lifecycle, timing, and the step loop.
    """

    def __init__(
        self,
        action_sink: ActionSink,
        observation_source: ObservationSource,
        reset_strategy: ResetStrategy,
        config: GameBridgeConfig,
    ):
        self.action_sink = action_sink
        self.observation_source = observation_source
        self.reset_strategy = reset_strategy
        self.config = config
        self._timer = StepTimer(config.timing)
        self._connected = False
        self._action_space = action_sink.info().action_space
        self._observation_space = observation_source.info().observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self.action_sink.connect()
        self.observation_source.connect()
        self._connected = True

    def disconnect(self) -> None:
        self.action_sink.disconnect()
        self.observation_source.disconnect()
        self._connected = False

    def reset(self) -> np.ndarray:
        self.action_sink.reset()
        self.reset_strategy.reset()
        self._timer.reset()
        return self.observation_source.read()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        self.action_sink.send(action)
        self._timer.wait()
        obs = self.observation_source.read()
        terminal = self.observation_source.is_terminal()
        return obs, terminal

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
