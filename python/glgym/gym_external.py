from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

from bridges.core.bridge import GameBridge


class ExternalGameGym(gym.Env):
    """Gymnasium wrapper for any GameBridge.

    Peer to BaseGhostLobbyGym -- both produce standard gym.Env objects
    that the training pipeline (PPOTrainer, CurriculumRunner) can consume
    interchangeably.

    BaseGhostLobbyGym wraps the Rust engine (PyO3).
    ExternalGameGym wraps any external game via a GameBridge.

    Attribute names match BaseGhostLobbyGym (current_step, episode_reward,
    max_steps, phase) so callbacks and logging code work on either.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bridge: GameBridge,
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
        max_steps: int = 2048,
        phase: int | None = None,
        auto_connect: bool = True,
    ):
        super().__init__()
        self.bridge = bridge
        self.reward_fn = reward_fn or (lambda prev, act, cur: 0.0)
        self.max_steps = max_steps
        self.phase = phase

        self.observation_space = bridge.observation_space
        self.action_space = bridge.action_space
        self.feature_index = bridge.feature_index

        self.current_step = 0
        self.episode_reward = 0.0
        self._prev_obs: np.ndarray | None = None

        if auto_connect:
            self.bridge.connect()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0

        obs = self.bridge.reset()
        self._prev_obs = obs.copy()
        return obs, {}

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        obs, terminal = self.bridge.step(action_arr)

        reward = self.reward_fn(self._prev_obs, action_arr, obs)
        self._prev_obs = obs.copy()
        self.current_step += 1
        self.episode_reward += reward

        truncated = self.current_step >= self.max_steps

        info = {}
        if terminal or truncated:
            info["episode_reward"] = self.episode_reward
            info["episode_steps"] = self.current_step

        return obs, reward, terminal, truncated, info

    def close(self):
        self.bridge.disconnect()
