"""Multi-input wrapper: splits flat obs into Dict with screen image + RAM vector.

SB3's MultiInputPolicy applies a CNN to the screen and an MLP to the RAM,
then concatenates before the policy head. This gives the agent spatial
awareness (walls, paths, NPCs) alongside game state (HP, level, battle).

Usage:
    bridge = make_pokemon_gold_bridge(..., include_screen=True, screen_downscale=4)
    env = ExternalGameGym(bridge=bridge, reward_fn=reward_fn)
    env = MultiInputWrapper(env, n_ram=25, screen_h=36, screen_w=40)
    # obs = {"screen": (36, 40, 1), "ram": (25,)}
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class MultiInputWrapper(gym.ObservationWrapper):
    """Splits a flat observation into a Dict with 'screen' and 'ram' keys.

    Expects the flat obs to be [ram_features..., screen_pixels...] as
    produced by PyBoyObservationSource with include_screen=True.
    """

    def __init__(
        self,
        env: gym.Env,
        n_ram: int,
        screen_h: int = 36,
        screen_w: int = 40,
    ):
        super().__init__(env)
        self._n_ram = n_ram
        self._screen_h = screen_h
        self._screen_w = screen_w
        self._n_screen = screen_h * screen_w

        expected_dim = n_ram + self._n_screen
        actual_dim = env.observation_space.shape[0]
        assert actual_dim == expected_dim, (
            f"Obs dim mismatch: env has {actual_dim}, expected {n_ram} RAM + "
            f"{self._n_screen} screen = {expected_dim}"
        )

        self.observation_space = gym.spaces.Dict({
            "screen": gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(screen_h, screen_w, 1),
                dtype=np.float32,
            ),
            "ram": gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(n_ram,),
                dtype=np.float32,
            ),
        })

    def observation(self, obs: np.ndarray) -> dict:
        ram = obs[:self._n_ram]
        screen = obs[self._n_ram:].reshape(self._screen_h, self._screen_w, 1)
        return {"screen": screen, "ram": ram}
