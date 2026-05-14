"""Anti-loop wrapper: penalizes repetitive action patterns.

PokeRL (2026) showed looping dropped from 41.2% to 4.7% of episodes
with anti-loop penalties. This wrapper detects two patterns:

1. Button spam: same button pressed N+ times in a row
2. Action cycling: short repeating pattern (e.g., left-right-left-right)

Penalties are applied as negative reward adjustments.

Usage:
    env = ExternalGameGym(bridge=bridge, reward_fn=reward_fn)
    env = AntiLoopWrapper(env, spam_threshold=3, cycle_window=8)
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class AntiLoopWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        spam_threshold: int = 3,
        spam_penalty: float = -0.1,
        heavy_spam_threshold: int = 6,
        heavy_spam_penalty: float = -0.3,
        cycle_window: int = 8,
        cycle_penalty: float = -0.15,
        idle_threshold: int = 10,
        idle_penalty: float = -0.2,
    ):
        super().__init__(env)
        self._spam_threshold = spam_threshold
        self._spam_penalty = spam_penalty
        self._heavy_spam_threshold = heavy_spam_threshold
        self._heavy_spam_penalty = heavy_spam_penalty
        self._cycle_window = cycle_window
        self._cycle_penalty = cycle_penalty
        self._idle_threshold = idle_threshold
        self._idle_penalty = idle_penalty

        self._action_history: deque[tuple] = deque(maxlen=cycle_window)
        self._consecutive_same = 0
        self._last_action: tuple | None = None
        self._idle_count = 0

    def reset(self, **kwargs):
        self._action_history.clear()
        self._consecutive_same = 0
        self._last_action = None
        self._idle_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        action_tuple = tuple(np.asarray(action).flat)
        penalty = 0.0

        # Spam detection
        if action_tuple == self._last_action:
            self._consecutive_same += 1
        else:
            self._consecutive_same = 1
        self._last_action = action_tuple

        if self._consecutive_same >= self._heavy_spam_threshold:
            penalty += self._heavy_spam_penalty
        elif self._consecutive_same >= self._spam_threshold:
            penalty += self._spam_penalty

        # Idle detection (all zeros = no buttons)
        if all(v == 0 for v in action_tuple):
            self._idle_count += 1
            if self._idle_count >= self._idle_threshold:
                penalty += self._idle_penalty
        else:
            self._idle_count = 0

        # Cycle detection: check if second half of window repeats first half
        self._action_history.append(action_tuple)
        if len(self._action_history) == self._cycle_window:
            half = self._cycle_window // 2
            hist = list(self._action_history)
            if hist[:half] == hist[half:]:
                penalty += self._cycle_penalty

        reward += penalty
        info["loop_penalty"] = penalty
        return obs, reward, terminated, truncated, info
