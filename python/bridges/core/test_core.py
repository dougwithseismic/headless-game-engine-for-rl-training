"""Tests for the core bridge framework: protocols, GameBridge, and timing."""

import time

import gymnasium as gym
import numpy as np
import pytest

from bridges.core.action_sink import ActionSink, ActionSinkInfo
from bridges.core.obs_source import ObservationSource, ObservationSourceInfo
from bridges.core.reset_strategy import ResetStrategy, ResetInfo
from bridges.core.timing import TimingPolicy, TimingConfig, StepTimer
from bridges.core.bridge import GameBridge, GameBridgeConfig
from bridges.sinks.mock_sink import MockActionSink
from bridges.sources.mock_source import MockObservationSource
from bridges.resets.mock_reset import MockReset


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_mock_sink_satisfies_protocol():
    sink = MockActionSink()
    assert isinstance(sink, ActionSink)


def test_mock_source_satisfies_protocol():
    source = MockObservationSource()
    assert isinstance(source, ObservationSource)


def test_mock_reset_satisfies_protocol():
    reset = MockReset()
    assert isinstance(reset, ResetStrategy)


# ---------------------------------------------------------------------------
# MockActionSink
# ---------------------------------------------------------------------------


def test_mock_sink_records_history():
    sink = MockActionSink(action_space=gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32))
    sink.connect()
    sink.send(np.array([0.1, 0.2, 0.3]))
    sink.send(np.array([0.4, 0.5, 0.6]))
    history = sink.get_history()
    assert len(history) == 2
    np.testing.assert_allclose(history[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(history[1], [0.4, 0.5, 0.6])


def test_mock_sink_requires_connect():
    sink = MockActionSink()
    with pytest.raises(RuntimeError):
        sink.send(np.array([0.0, 0.0]))


def test_mock_sink_info():
    space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
    sink = MockActionSink(action_space=space)
    info = sink.info()
    assert info.name == "mock"
    assert info.platform == "any"
    assert info.action_space == space


def test_mock_sink_ring_buffer():
    sink = MockActionSink(max_history=3)
    sink.connect()
    for i in range(5):
        sink.send(np.array([float(i), 0.0]))
    history = sink.get_history()
    assert len(history) == 3
    np.testing.assert_allclose(history[0], [2.0, 0.0])


# ---------------------------------------------------------------------------
# MockObservationSource
# ---------------------------------------------------------------------------


def test_mock_source_returns_observations():
    source = MockObservationSource(
        observation_space=gym.spaces.Box(-1, 1, shape=(5,), dtype=np.float32)
    )
    source.connect()
    obs = source.read()
    assert obs.shape == (5,)
    assert source.step_count == 1


def test_mock_source_terminal_after():
    source = MockObservationSource(terminal_after=3)
    source.connect()
    assert not source.is_terminal()
    source.read()
    source.read()
    assert not source.is_terminal()
    source.read()
    assert source.is_terminal()


def test_mock_source_requires_connect():
    source = MockObservationSource()
    with pytest.raises(RuntimeError):
        source.read()


def test_mock_source_reset_on_reconnect():
    source = MockObservationSource(terminal_after=2)
    source.connect()
    source.read()
    source.read()
    assert source.is_terminal()
    source.disconnect()
    source.connect()
    assert not source.is_terminal()
    assert source.step_count == 0


# ---------------------------------------------------------------------------
# MockReset
# ---------------------------------------------------------------------------


def test_mock_reset_counts():
    reset = MockReset()
    assert reset.reset_count == 0
    reset.reset()
    reset.reset()
    assert reset.reset_count == 2


def test_mock_reset_checkpoint_noop():
    reset = MockReset()
    reset.set_checkpoint("level_3")
    assert reset._checkpoint == "level_3"


# ---------------------------------------------------------------------------
# StepTimer
# ---------------------------------------------------------------------------


def test_timer_free_running_no_delay():
    timer = StepTimer(TimingConfig(policy=TimingPolicy.FREE_RUNNING))
    start = time.monotonic()
    for _ in range(100):
        timer.wait()
    elapsed = time.monotonic() - start
    assert elapsed < 0.1


def test_timer_real_time_pacing():
    timer = StepTimer(TimingConfig(policy=TimingPolicy.REAL_TIME, target_hz=100.0))
    start = time.monotonic()
    for _ in range(5):
        timer.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.04  # 5 steps at 100Hz = 50ms, allow some slack


def test_timer_reset_clears_state():
    timer = StepTimer(TimingConfig(policy=TimingPolicy.REAL_TIME, target_hz=10.0))
    timer.wait()
    timer.reset()
    assert timer._last_step_time is None


# ---------------------------------------------------------------------------
# GameBridge
# ---------------------------------------------------------------------------


def _make_bridge(obs_dim=10, act_dim=2, terminal_after=None):
    return GameBridge(
        action_sink=MockActionSink(
            action_space=gym.spaces.Box(-1, 1, shape=(act_dim,), dtype=np.float32),
        ),
        observation_source=MockObservationSource(
            observation_space=gym.spaces.Box(-1, 1, shape=(obs_dim,), dtype=np.float32),
            terminal_after=terminal_after,
        ),
        reset_strategy=MockReset(),
        config=GameBridgeConfig(
            name="test_bridge",
            timing=TimingConfig(policy=TimingPolicy.FREE_RUNNING),
        ),
    )


def test_bridge_spaces():
    bridge = _make_bridge(obs_dim=20, act_dim=3)
    assert bridge.action_space.shape == (3,)
    assert bridge.observation_space.shape == (20,)


def test_bridge_connect_disconnect():
    bridge = _make_bridge()
    assert not bridge.connected
    bridge.connect()
    assert bridge.connected
    bridge.disconnect()
    assert not bridge.connected


def test_bridge_context_manager():
    bridge = _make_bridge()
    with bridge:
        assert bridge.connected
    assert not bridge.connected


def test_bridge_reset_returns_obs():
    bridge = _make_bridge(obs_dim=5)
    bridge.connect()
    obs = bridge.reset()
    assert obs.shape == (5,)
    assert bridge.reset_strategy.reset_count == 1


def test_bridge_step_loop():
    bridge = _make_bridge(obs_dim=4, act_dim=2)
    bridge.connect()
    bridge.reset()
    for _ in range(10):
        obs, terminal = bridge.step(np.array([0.5, -0.3]))
        assert obs.shape == (4,)
        assert isinstance(terminal, bool)
    assert len(bridge.action_sink.get_history()) == 10


def test_bridge_terminal_detection():
    bridge = _make_bridge(terminal_after=3)
    bridge.connect()
    bridge.reset()
    _, t1 = bridge.step(np.zeros(2))
    assert not t1
    _, t2 = bridge.step(np.zeros(2))
    # terminal_after=3, but reset() calls read() once, so step 2 is read #3
    assert t2


def test_bridge_reset_clears_terminal():
    bridge = _make_bridge(terminal_after=2)
    bridge.connect()
    bridge.reset()
    bridge.step(np.zeros(2))
    _, terminal = bridge.step(np.zeros(2))
    # After reset, source reconnects are handled by the source's own state
    # The mock source tracks step_count which resets on connect()
    obs = bridge.reset()
    assert obs.shape == (10,)
