"""Tests for game profiles and ExternalGameGym integration."""

import numpy as np
import pytest

from bridges.profiles.template import make_template_bridge
from bridges.core.bridge import GameBridge
from glgym.gym_external import ExternalGameGym


# ---------------------------------------------------------------------------
# Template profile
# ---------------------------------------------------------------------------


def test_template_produces_bridge():
    bridge = make_template_bridge(obs_dim=8, act_dim=3)
    assert isinstance(bridge, GameBridge)
    assert bridge.observation_space.shape == (8,)
    assert bridge.action_space.shape == (3,)


def test_template_bridge_step_loop():
    bridge = make_template_bridge(obs_dim=5, act_dim=2)
    bridge.connect()
    obs = bridge.reset()
    assert obs.shape == (5,)
    for _ in range(20):
        obs, terminal = bridge.step(np.array([0.1, -0.2]))
        assert obs.shape == (5,)
    bridge.disconnect()


# ---------------------------------------------------------------------------
# ExternalGameGym integration
# ---------------------------------------------------------------------------


def test_gym_external_reset():
    bridge = make_template_bridge(obs_dim=10, act_dim=2)
    env = ExternalGameGym(bridge=bridge, max_steps=100)
    obs, info = env.reset()
    assert obs.shape == (10,)
    assert isinstance(info, dict)
    env.close()


def test_gym_external_step():
    bridge = make_template_bridge(obs_dim=10, act_dim=2)
    env = ExternalGameGym(bridge=bridge, max_steps=100)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_gym_external_truncation():
    bridge = make_template_bridge(obs_dim=4, act_dim=1, max_steps=5)
    env = ExternalGameGym(bridge=bridge, max_steps=5)
    env.reset()
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert truncated
    assert "episode_steps" in info
    assert info["episode_steps"] == 5
    env.close()


def test_gym_external_terminal():
    bridge = make_template_bridge(obs_dim=4, act_dim=1, terminal_after=3)
    env = ExternalGameGym(bridge=bridge, max_steps=100)
    env.reset()
    # reset() calls read() once, so terminal_after=3 means 2 more steps
    env.step(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert terminated
    assert "episode_reward" in info
    env.close()


def test_gym_external_custom_reward():
    def reward_fn(prev_obs, action, obs):
        return float(np.sum(obs) - np.sum(prev_obs))

    bridge = make_template_bridge(obs_dim=4, act_dim=1)
    env = ExternalGameGym(bridge=bridge, reward_fn=reward_fn, max_steps=100)
    env.reset()
    obs, reward, _, _, _ = env.step(env.action_space.sample())
    assert isinstance(reward, float)
    env.close()


def test_gym_external_episode_loop():
    """Full episode loop: reset -> step until done -> reset -> step."""
    bridge = make_template_bridge(obs_dim=6, act_dim=2, max_steps=10)
    env = ExternalGameGym(bridge=bridge, max_steps=10)

    for episode in range(3):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
        assert steps <= 10

    env.close()


def test_gym_external_observation_space_matches():
    bridge = make_template_bridge(obs_dim=12, act_dim=4)
    env = ExternalGameGym(bridge=bridge, max_steps=50)
    assert env.observation_space.shape == (12,)
    assert env.action_space.shape == (4,)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    env.close()
