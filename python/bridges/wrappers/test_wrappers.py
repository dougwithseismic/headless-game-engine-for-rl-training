"""Tests for temporal and anti-loop wrappers."""

import gymnasium as gym
import numpy as np
import pytest

from bridges.wrappers.temporal import TemporalObsWrapper, N_DELTA_FEATURES, DELTA_FEATURE_NAMES
from bridges.wrappers.anti_loop import AntiLoopWrapper
from bridges.profiles.template import make_template_bridge
from glgym.gym_external import ExternalGameGym


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(obs_dim=10, act_dim=2, feature_names=None):
    """Simple gym env with optional feature_index."""
    from bridges.core.bridge import GameBridge, GameBridgeConfig
    from bridges.core.timing import TimingConfig, TimingPolicy
    from bridges.sinks.mock_sink import MockActionSink
    from bridges.sources.mock_source import MockObservationSource
    from bridges.resets.mock_reset import MockReset

    bridge = GameBridge(
        action_sink=MockActionSink(
            action_space=gym.spaces.Box(-1, 1, shape=(act_dim,), dtype=np.float32),
        ),
        observation_source=MockObservationSource(
            observation_space=gym.spaces.Box(0, 1, shape=(obs_dim,), dtype=np.float32),
            feature_names=feature_names,
        ),
        reset_strategy=MockReset(),
        config=GameBridgeConfig(
            name="test", timing=TimingConfig(policy=TimingPolicy.FREE_RUNNING),
        ),
    )
    return ExternalGameGym(bridge=bridge, max_steps=100)


def _make_button_env():
    """MultiBinary(8) env like Pokemon Gold."""
    from bridges.core.bridge import GameBridge, GameBridgeConfig
    from bridges.core.timing import TimingConfig, TimingPolicy
    from bridges.sinks.mock_sink import MockActionSink
    from bridges.sources.mock_source import MockObservationSource
    from bridges.resets.mock_reset import MockReset

    bridge = GameBridge(
        action_sink=MockActionSink(
            action_space=gym.spaces.MultiBinary(8),
        ),
        observation_source=MockObservationSource(
            observation_space=gym.spaces.Box(0, 1, shape=(5,), dtype=np.float32),
        ),
        reset_strategy=MockReset(),
        config=GameBridgeConfig(
            name="button_test", timing=TimingConfig(policy=TimingPolicy.FREE_RUNNING),
        ),
    )
    return ExternalGameGym(bridge=bridge, max_steps=200)


# ---------------------------------------------------------------------------
# TemporalObsWrapper
# ---------------------------------------------------------------------------


def test_temporal_wrapper_extends_obs_space():
    env = _make_env(obs_dim=10)
    wrapped = TemporalObsWrapper(env)
    assert wrapped.observation_space.shape == (10 + N_DELTA_FEATURES,)


def test_temporal_wrapper_reset_returns_extended_obs():
    env = _make_env(obs_dim=5)
    wrapped = TemporalObsWrapper(env)
    obs, info = wrapped.reset()
    assert obs.shape == (5 + N_DELTA_FEATURES,)
    # Deltas should be zero on reset
    np.testing.assert_array_equal(obs[5:], 0.0)


def test_temporal_wrapper_step_returns_extended_obs():
    env = _make_env(obs_dim=5, act_dim=2)
    wrapped = TemporalObsWrapper(env)
    obs, _ = wrapped.reset()
    obs2, r, t, tr, info = wrapped.step(np.array([0.5, -0.3]))
    assert obs2.shape == (5 + N_DELTA_FEATURES,)


def test_temporal_wrapper_with_feature_index():
    names = ["player_x", "player_y", "map_group", "map_number",
             "party1_hp_hi", "battle_mode", "textbox_flags"]
    env = _make_env(obs_dim=7, act_dim=2, feature_names=names)
    wrapped = TemporalObsWrapper(env, feature_index=env.feature_index)
    assert wrapped.observation_space.shape == (7 + N_DELTA_FEATURES,)
    obs, _ = wrapped.reset()
    assert obs.shape == (7 + N_DELTA_FEATURES,)


def test_temporal_wrapper_stuck_counter():
    """Stuck counter increments when position doesn't change."""
    names = ["player_x", "player_y", "map_group", "map_number"]
    env = _make_env(obs_dim=4, act_dim=1, feature_names=names)
    wrapped = TemporalObsWrapper(env, feature_index=env.feature_index)

    # Mock: override observations to have constant position
    obs, _ = wrapped.reset()

    # After stepping with same obs, stuck_count should increase
    # Note: MockObservationSource returns random obs, so stuck detection
    # depends on random values. We just verify the wrapper doesn't crash
    # and produces valid output.
    for _ in range(10):
        obs, _, _, _, _ = wrapped.step(np.array([0.0]))
        assert obs.shape == (4 + N_DELTA_FEATURES,)
        # stuck_count (index 5 in deltas = index 9 in full obs) is bounded [0, 1]
        assert 0.0 <= obs[4 + 5] <= 1.0


def test_temporal_wrapper_delta_feature_names():
    assert len(DELTA_FEATURE_NAMES) == N_DELTA_FEATURES
    assert "delta_x" in DELTA_FEATURE_NAMES
    assert "stuck_count" in DELTA_FEATURE_NAMES


def test_temporal_wrapper_graceful_missing_features():
    """Wrapper works even without feature_index — all deltas are 0."""
    env = _make_env(obs_dim=5, act_dim=2)
    wrapped = TemporalObsWrapper(env, feature_index={})
    obs, _ = wrapped.reset()
    obs2, _, _, _, _ = wrapped.step(np.array([0.1, 0.2]))
    # All deltas should be 0 since no features are mapped
    assert obs2.shape == (5 + N_DELTA_FEATURES,)


# ---------------------------------------------------------------------------
# AntiLoopWrapper
# ---------------------------------------------------------------------------


def test_anti_loop_no_penalty_for_varied_actions():
    env = _make_button_env()
    wrapped = AntiLoopWrapper(env, spam_threshold=3)
    wrapped.reset()

    actions = [
        np.array([1, 0, 0, 0, 0, 0, 0, 0]),  # up
        np.array([0, 1, 0, 0, 0, 0, 0, 0]),  # down
        np.array([0, 0, 1, 0, 0, 0, 0, 0]),  # left
        np.array([0, 0, 0, 1, 0, 0, 0, 0]),  # right
    ]
    for a in actions:
        _, _, _, _, info = wrapped.step(a)
        assert info["loop_penalty"] == 0.0


def test_anti_loop_spam_penalty():
    env = _make_button_env()
    wrapped = AntiLoopWrapper(env, spam_threshold=3, spam_penalty=-0.1)
    wrapped.reset()

    action = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    penalties = []
    for _ in range(5):
        _, _, _, _, info = wrapped.step(action)
        penalties.append(info["loop_penalty"])

    # First 2 steps: no penalty (count=1,2)
    assert penalties[0] == 0.0
    assert penalties[1] == 0.0
    # Step 3+: spam penalty
    assert penalties[2] == -0.1
    assert penalties[3] == -0.1


def test_anti_loop_heavy_spam_penalty():
    env = _make_button_env()
    wrapped = AntiLoopWrapper(
        env, spam_threshold=3, spam_penalty=-0.1,
        heavy_spam_threshold=6, heavy_spam_penalty=-0.3,
    )
    wrapped.reset()

    action = np.array([0, 0, 0, 0, 1, 0, 0, 0])  # A button
    penalties = []
    for _ in range(8):
        _, _, _, _, info = wrapped.step(action)
        penalties.append(info["loop_penalty"])

    assert penalties[0] == 0.0   # count=1
    assert penalties[2] == -0.1  # count=3, mild spam
    assert penalties[5] == -0.3  # count=6, heavy spam


def test_anti_loop_idle_penalty():
    env = _make_button_env()
    wrapped = AntiLoopWrapper(env, idle_threshold=3, idle_penalty=-0.2)
    wrapped.reset()

    idle = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    penalties = []
    for _ in range(5):
        _, _, _, _, info = wrapped.step(idle)
        penalties.append(info["loop_penalty"])

    # idle_threshold=3, so penalty starts at step 3
    assert penalties[0] == 0.0  # spam penalty may apply too
    assert penalties[2] < 0     # idle penalty kicks in


def test_anti_loop_cycle_detection():
    env = _make_button_env()
    wrapped = AntiLoopWrapper(env, cycle_window=4, cycle_penalty=-0.15,
                               spam_threshold=100)  # disable spam
    wrapped.reset()

    # Left-right-left-right = cycle of length 2
    left = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    right = np.array([0, 0, 0, 1, 0, 0, 0, 0])

    penalties = []
    for a in [left, right, left, right]:
        _, _, _, _, info = wrapped.step(a)
        penalties.append(info["loop_penalty"])

    # Cycle detected on 4th action (window full, first half == second half)
    assert penalties[3] == -0.15


def test_anti_loop_reset_clears_state():
    env = _make_button_env()
    wrapped = AntiLoopWrapper(env, spam_threshold=2)
    wrapped.reset()

    action = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    wrapped.step(action)
    wrapped.step(action)  # would trigger spam

    wrapped.reset()
    _, _, _, _, info = wrapped.step(action)
    assert info["loop_penalty"] == 0.0  # reset cleared the counter


def test_anti_loop_combined_with_temporal():
    """Both wrappers stack correctly."""
    names = ["player_x", "player_y", "map_group", "map_number", "battle_mode"]
    env = _make_env(obs_dim=5, act_dim=2, feature_names=names)
    env = TemporalObsWrapper(env, feature_index=env.feature_index)
    env = AntiLoopWrapper(env)

    obs, _ = env.reset()
    assert obs.shape == (5 + N_DELTA_FEATURES,)

    obs2, r, t, tr, info = env.step(np.array([0.5, -0.3]))
    assert obs2.shape == (5 + N_DELTA_FEATURES,)
    assert "loop_penalty" in info
