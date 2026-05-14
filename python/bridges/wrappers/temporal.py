"""Temporal observation wrapper: adds computed delta features to observations.

Appends a fixed set of derived temporal signals to the base observation:
- Velocity: dx, dy per step (position change)
- HP delta: change in HP since last step
- Battle transition: just entered/exited battle
- Stuck counter: consecutive steps at same (map, x, y)
- Menu transition: just entered/exited menu

These give the agent temporal context without requiring frame stacking
or recurrent policies. Can be combined with VecFrameStack for both
raw history and pre-computed signals.

Usage:
    env = ExternalGameGym(bridge=bridge, reward_fn=reward_fn)
    env = TemporalObsWrapper(env, feature_index=bridge.feature_index)
    # obs space grows by N_DELTA_FEATURES (8 features)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


N_DELTA_FEATURES = 8

DELTA_FEATURE_NAMES = [
    "delta_x",
    "delta_y",
    "delta_hp",
    "battle_entered",
    "battle_exited",
    "stuck_count",
    "menu_entered",
    "menu_exited",
]


class TemporalObsWrapper(gym.ObservationWrapper):
    """Appends temporal delta features to observations.

    Requires a feature_index mapping observation names to indices.
    Gracefully degrades: if a feature name isn't in the index,
    the corresponding delta is always 0.
    """

    def __init__(
        self,
        env: gym.Env,
        feature_index: dict[str, int] | None = None,
        max_stuck_norm: float = 50.0,
    ):
        super().__init__(env)
        self.idx = feature_index or getattr(env, "feature_index", {})
        self._max_stuck_norm = max_stuck_norm

        base_shape = env.observation_space.shape
        assert base_shape is not None and len(base_shape) == 1
        new_dim = base_shape[0] + N_DELTA_FEATURES

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(new_dim,), dtype=np.float32,
        )

        self._prev_obs: np.ndarray | None = None
        self._stuck_count = 0
        self._prev_pos: tuple[float, float, float, float] | None = None
        self._prev_battle = 0.0
        self._prev_menu = 0.0
        self._is_first_obs = True

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs.copy()
        self._stuck_count = 0
        self._prev_pos = self._get_pos(obs)
        self._prev_battle = self._get_val(obs, "battle_mode")
        self._prev_menu = self._get_val(obs, "textbox_flags")
        self._is_first_obs = True
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        deltas = np.zeros(N_DELTA_FEATURES, dtype=np.float32)

        if self._prev_obs is not None and not self._is_first_obs:
            # Velocity
            deltas[0] = self._get_val(obs, "player_x") - self._get_val(self._prev_obs, "player_x")
            deltas[1] = self._get_val(obs, "player_y") - self._get_val(self._prev_obs, "player_y")

            # HP delta (using high byte — coarse but fast)
            hp_key = "party1_hp_hi" if "party1_hp_hi" in self.idx else "party_hp_1"
            deltas[2] = self._get_val(obs, hp_key) - self._get_val(self._prev_obs, hp_key)

            # Battle transitions
            battle = self._get_val(obs, "battle_mode")
            deltas[3] = 1.0 if battle > 0 and self._prev_battle == 0 else 0.0
            deltas[4] = 1.0 if battle == 0 and self._prev_battle > 0 else 0.0
            self._prev_battle = battle

            # Stuck counter
            pos = self._get_pos(obs)
            if pos == self._prev_pos:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
            self._prev_pos = pos
            deltas[5] = min(self._stuck_count / self._max_stuck_norm, 1.0)

            # Menu transitions
            menu = self._get_val(obs, "textbox_flags")
            deltas[6] = 1.0 if menu > 0 and self._prev_menu == 0 else 0.0
            deltas[7] = 1.0 if menu == 0 and self._prev_menu > 0 else 0.0
            self._prev_menu = menu

        self._prev_obs = obs.copy()
        self._is_first_obs = False
        return np.concatenate([obs, deltas])

    def _get_val(self, obs: np.ndarray, name: str) -> float:
        if name in self.idx:
            return float(obs[self.idx[name]])
        return 0.0

    def _get_pos(self, obs: np.ndarray) -> tuple[float, float, float, float]:
        return (
            self._get_val(obs, "map_group"),
            self._get_val(obs, "map_number"),
            self._get_val(obs, "player_x"),
            self._get_val(obs, "player_y"),
        )
