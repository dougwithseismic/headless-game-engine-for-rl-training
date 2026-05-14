"""Template game profile. Copy and customize for a new game.

Usage:
    from bridges.profiles.template import make_template_bridge
    from glgym.gym_external import ExternalGameGym

    bridge = make_template_bridge(obs_dim=10, act_dim=2)
    env = ExternalGameGym(bridge=bridge, max_steps=100)
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    env.close()
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from bridges.core.bridge import GameBridge, GameBridgeConfig
from bridges.core.timing import TimingConfig, TimingPolicy
from bridges.sinks.mock_sink import MockActionSink
from bridges.sources.mock_source import MockObservationSource
from bridges.resets.mock_reset import MockReset


def make_template_bridge(
    obs_dim: int = 10,
    act_dim: int = 2,
    terminal_after: int | None = None,
) -> GameBridge:
    """Template game bridge with mock components.

    Replace MockActionSink / MockObservationSource / MockReset
    with real implementations for your target game.
    """
    return GameBridge(
        action_sink=MockActionSink(
            action_space=gym.spaces.Box(
                low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32,
            ),
        ),
        observation_source=MockObservationSource(
            observation_space=gym.spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
            ),
            terminal_after=terminal_after,
        ),
        reset_strategy=MockReset(),
        config=GameBridgeConfig(
            name="template_game",
            timing=TimingConfig(policy=TimingPolicy.FREE_RUNNING),
        ),
    )
