"""Tests for PyBoy bridge components.

Tests marked with @pytest.mark.rom require a Pokemon Red ROM at
the path specified by the POKEMON_ROM environment variable.
Run with: POKEMON_ROM=/path/to/pokemon_red.gb pytest -v -m rom
"""

import os

import numpy as np
import pytest

from bridges.emulators.pyboy_host import PyBoyHost, GAMEBOY_BUTTONS
from bridges.sinks.pyboy_sink import PyBoyActionSink
from bridges.sources.pyboy_source import PyBoyObservationSource, RAMFeature
from bridges.resets.pyboy_reset import PyBoyReset
from bridges.core.action_sink import ActionSink
from bridges.core.obs_source import ObservationSource
from bridges.core.reset_strategy import ResetStrategy


ROM_PATH = os.environ.get("POKEMON_ROM", "")
has_rom = os.path.isfile(ROM_PATH) if ROM_PATH else False
rom_required = pytest.mark.skipif(not has_rom, reason="POKEMON_ROM env var not set or file missing")


# ---------------------------------------------------------------------------
# Protocol compliance (no ROM needed)
# ---------------------------------------------------------------------------


def test_pyboy_sink_satisfies_protocol():
    host = PyBoyHost("dummy.gb")
    sink = PyBoyActionSink(host)
    assert isinstance(sink, ActionSink)


def test_pyboy_source_satisfies_protocol():
    host = PyBoyHost("dummy.gb")
    source = PyBoyObservationSource(host)
    assert isinstance(source, ObservationSource)


def test_pyboy_reset_satisfies_protocol():
    host = PyBoyHost("dummy.gb")
    reset = PyBoyReset(host)
    assert isinstance(reset, ResetStrategy)


def test_sink_info():
    host = PyBoyHost("dummy.gb")
    sink = PyBoyActionSink(host)
    info = sink.info()
    assert info.name == "pyboy"
    assert info.platform == "any"
    assert info.supports_discrete is True


def test_source_info_ram_only():
    host = PyBoyHost("dummy.gb")
    features = [RAMFeature("x", 0x1234), RAMFeature("y", 0x1235)]
    source = PyBoyObservationSource(host, ram_features=features)
    info = source.info()
    assert info.observation_space.shape == (2,)


def test_source_info_ram_and_screen():
    host = PyBoyHost("dummy.gb")
    features = [RAMFeature("x", 0x1234)]
    source = PyBoyObservationSource(host, ram_features=features, include_screen=True, screen_downscale=4)
    info = source.info()
    # 1 RAM feature + (144/4 * 160/4) = 1 + 1440 = 1441
    assert info.observation_space.shape == (1441,)


def test_reset_info():
    host = PyBoyHost("dummy.gb")
    reset = PyBoyReset(host)
    info = reset.info()
    assert info.is_instant is True
    assert info.supports_checkpoints is True


def test_gameboy_buttons_are_eight():
    assert len(GAMEBOY_BUTTONS) == 8


# ---------------------------------------------------------------------------
# Live tests (need ROM)
# ---------------------------------------------------------------------------


@rom_required
def test_host_start_stop():
    host = PyBoyHost(ROM_PATH, headless=True, speed=0)
    host.start()
    assert host.started
    host.tick(60, render=False)
    assert host.frame_count >= 60
    host.stop()
    assert not host.started


@rom_required
def test_host_context_manager():
    with PyBoyHost(ROM_PATH, headless=True, speed=0) as host:
        assert host.started
        host.tick(10, render=False)
    assert not host.started


@rom_required
def test_host_save_load_state():
    with PyBoyHost(ROM_PATH, headless=True, speed=0) as host:
        host.tick(60, render=False)
        host.save_state_to_buffer("test")
        frame_before = host.frame_count
        host.tick(60, render=False)
        assert host.frame_count > frame_before
        host.load_state_from_buffer("test")


@rom_required
def test_host_memory_read():
    with PyBoyHost(ROM_PATH, headless=True, speed=0) as host:
        host.tick(60, render=False)
        val = host.read_memory(0xD362)
        assert isinstance(val, int)
        assert 0 <= val <= 255


@rom_required
def test_host_screen():
    with PyBoyHost(ROM_PATH, headless=True, speed=0) as host:
        host.tick(60, render=True)
        screen = host.screen_ndarray()
        assert screen.ndim >= 2
        assert screen.shape[0] == 144
        assert screen.shape[1] == 160


@rom_required
def test_sink_send_buttons():
    host = PyBoyHost(ROM_PATH, headless=True, speed=0)
    sink = PyBoyActionSink(host)
    sink.connect()
    action = np.zeros(8, dtype=np.int8)
    action[4] = 1  # press A
    sink.send(action)
    assert "a" in sink._held
    action[4] = 0
    sink.send(action)
    assert "a" not in sink._held
    host.stop()


@rom_required
def test_source_read_ram():
    host = PyBoyHost(ROM_PATH, headless=True, speed=0)
    features = [RAMFeature("x", 0xD362), RAMFeature("y", 0xD361)]
    source = PyBoyObservationSource(host, ram_features=features, ticks_per_step=1, render=False)
    source.connect()
    obs = source.read()
    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)
    host.stop()


@rom_required
def test_source_read_with_screen():
    host = PyBoyHost(ROM_PATH, headless=True, speed=0)
    features = [RAMFeature("x", 0xD362)]
    source = PyBoyObservationSource(
        host, ram_features=features, include_screen=True, screen_downscale=4, ticks_per_step=1,
    )
    source.connect()
    obs = source.read()
    assert obs.shape == (1441,)  # 1 + 36*40
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)
    host.stop()


@rom_required
def test_reset_save_and_restore():
    host = PyBoyHost(ROM_PATH, headless=True, speed=0)
    host.start()
    host.tick(120, render=False)

    reset = PyBoyReset(host)

    # First reset saves initial state
    reset.reset()
    mem_after_save = host.read_memory(0xD362)

    # Advance the game
    host.tick(300, render=False)
    mem_after_advance = host.read_memory(0xD362)

    # Second reset restores
    reset.reset()
    mem_after_restore = host.read_memory(0xD362)
    assert mem_after_restore == mem_after_save

    host.stop()


@rom_required
def test_full_bridge_integration():
    """Full end-to-end: bridge with all three PyBoy components."""
    from bridges.core.bridge import GameBridge, GameBridgeConfig
    from bridges.core.timing import TimingConfig, TimingPolicy
    from glgym.gym_external import ExternalGameGym

    host = PyBoyHost(ROM_PATH, headless=True, speed=0)

    bridge = GameBridge(
        action_sink=PyBoyActionSink(host),
        observation_source=PyBoyObservationSource(
            host,
            ram_features=[RAMFeature("x", 0xD362), RAMFeature("y", 0xD361), RAMFeature("map", 0xD35E)],
            ticks_per_step=24,
            render=False,
        ),
        reset_strategy=PyBoyReset(host),
        config=GameBridgeConfig(
            name="pokemon_red_test",
            timing=TimingConfig(policy=TimingPolicy.FREE_RUNNING),
        ),
    )

    env = ExternalGameGym(bridge=bridge, max_steps=50)
    obs, info = env.reset()
    assert obs.shape == (3,)

    total_reward = 0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    host.stop()


@rom_required
def test_pokemon_red_profile():
    """Test the full profile factory."""
    from bridges.profiles.pokemon_red import make_pokemon_red_bridge, ExplorationReward
    from glgym.gym_external import ExternalGameGym

    bridge = make_pokemon_red_bridge(
        rom_path=ROM_PATH,
        headless=True,
        speed=0,
        ticks_per_step=24,
    )

    reward_fn = ExplorationReward(bridge.feature_index, new_tile_reward=1.0)
    env = ExternalGameGym(bridge=bridge, reward_fn=reward_fn, max_steps=100)

    obs, _ = env.reset()
    assert obs.shape[0] == 16  # 16 RAM features

    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
            reward_fn.reset()

    env.close()
