"""Tests for Pokemon telemetry wrapper."""

import gymnasium as gym
import numpy as np
import pytest

from bridges.wrappers.pokemon_telemetry import PokemonTelemetryWrapper, GameStats
from bridges.profiles.pokemon_gold import POKEMON_GOLD_RAM


NAMES = [f.name for f in POKEMON_GOLD_RAM]
IDX = {n: i for i, n in enumerate(NAMES)}
OBS_DIM = len(NAMES)


def _obs(**overrides) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    norms = {f.name: f.normalize_max for f in POKEMON_GOLD_RAM}
    for name, raw_val in overrides.items():
        if name in IDX:
            obs[IDX[name]] = raw_val / norms[name]
    return obs


# Baseline obs that represents "in-game, stable" state
BASE = dict(party_size=1, party1_level=5, party1_hp_hi=0, party1_hp_lo=20,
            party1_maxhp_hi=0, party1_maxhp_lo=20, map_group=1, map_number=1,
            player_x=5, player_y=5)


class FakeEnv(gym.Env):
    def __init__(self, obs_sequence: list[np.ndarray]):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 1, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.MultiBinary(8)
        self._obs_seq = obs_sequence
        self._step_idx = 0
        self.feature_index = IDX

    def reset(self, **kwargs):
        self._step_idx = 0
        return self._obs_seq[0].copy(), {}

    def step(self, action):
        self._step_idx = min(self._step_idx + 1, len(self._obs_seq) - 1)
        obs = self._obs_seq[self._step_idx].copy()
        terminated = self._step_idx >= len(self._obs_seq) - 1
        return obs, 0.0, terminated, False, {}


ACT = np.zeros(8)


def _b(**overrides):
    """Merge overrides into BASE."""
    return {**BASE, **overrides}


def test_telemetry_tracks_tiles():
    obs_seq = [
        _obs(**_b(player_x=5)),
        _obs(**_b(player_x=5)),
        _obs(**_b(player_x=6)),
        _obs(**_b(player_x=7)),
    ]
    env = PokemonTelemetryWrapper(FakeEnv(obs_seq), feature_index=IDX)
    env.reset()
    for _ in range(3):
        _, _, _, _, info = env.step(ACT)
    assert info["game_stats"]["tiles_visited"] == 3


def test_telemetry_detects_map_change():
    obs_seq = [
        _obs(**BASE),
        _obs(**BASE),
        _obs(**_b(map_group=1, map_number=2, player_x=0, player_y=0)),
        _obs(**_b(map_group=1, map_number=2, player_x=1, player_y=0)),
    ]
    env = PokemonTelemetryWrapper(FakeEnv(obs_seq), feature_index=IDX)
    env.reset()
    for _ in range(3):
        _, _, _, _, info = env.step(ACT)
    assert info["game_stats"]["maps_visited"] == 2
    events = [e for e in info["game_events"] if e["type"] == "map_enter"]
    assert len(events) == 1


def test_telemetry_detects_battle():
    obs_seq = [
        _obs(**_b(battle_mode=0)),
        _obs(**_b(battle_mode=0)),
        _obs(**_b(battle_mode=1, party1_hp_lo=15)),
        _obs(**_b(battle_mode=0, party1_hp_lo=10)),
    ]
    env = PokemonTelemetryWrapper(FakeEnv(obs_seq), feature_index=IDX)
    env.reset()
    for _ in range(3):
        _, _, _, _, info = env.step(ACT)
    assert info["game_stats"]["wild_battles"] == 1
    assert info["game_stats"]["battles_won"] == 1


def test_telemetry_detects_level_up():
    obs_seq = [
        _obs(**_b(party1_level=5)),
        _obs(**_b(party1_level=5)),
        _obs(**_b(party1_level=5)),
        _obs(**_b(party1_level=7)),
    ]
    env = PokemonTelemetryWrapper(FakeEnv(obs_seq), feature_index=IDX)
    env.reset()
    for _ in range(3):
        _, _, _, _, info = env.step(ACT)
    assert info["game_stats"]["level_ups"] == 2
    assert info["game_stats"]["max_level"] == 7


def test_telemetry_detects_badge():
    obs_seq = [
        _obs(**_b(johto_badges=0)),
        _obs(**_b(johto_badges=0)),
        _obs(**_b(johto_badges=0b00000001)),
    ]
    env = PokemonTelemetryWrapper(FakeEnv(obs_seq), feature_index=IDX)
    env.reset()
    for _ in range(2):
        _, _, _, _, info = env.step(ACT)
    assert info["game_stats"]["badges_earned"] == 1


def test_telemetry_detects_party_change():
    obs_seq = [
        _obs(**_b(party_size=1)),
        _obs(**_b(party_size=1)),
        _obs(**_b(party_size=2)),
    ]
    env = PokemonTelemetryWrapper(FakeEnv(obs_seq), feature_index=IDX)
    env.reset()
    for _ in range(2):
        _, _, _, _, info = env.step(ACT)
    assert info["game_stats"]["party_size_max"] == 2


def test_telemetry_battle_loss():
    obs_seq = [
        _obs(**_b(battle_mode=0, party1_hp_lo=20)),
        _obs(**_b(battle_mode=0, party1_hp_lo=20)),
        _obs(**_b(battle_mode=1, party1_hp_lo=5)),
        _obs(**_b(battle_mode=0, party1_hp_lo=0, party1_hp_hi=0)),
    ]
    env = PokemonTelemetryWrapper(FakeEnv(obs_seq), feature_index=IDX)
    env.reset()
    for _ in range(3):
        _, _, _, _, info = env.step(ACT)
    assert info["game_stats"]["battles_lost"] == 1


def test_telemetry_exploration_rate():
    obs_seq = [_obs(**_b(player_x=0))] + [
        _obs(**_b(player_x=i)) for i in range(12)
    ]
    env = PokemonTelemetryWrapper(FakeEnv(obs_seq), feature_index=IDX)
    env.reset()
    for _ in range(12):
        _, _, _, _, info = env.step(ACT)
    rate = info["game_stats"]["exploration_rate"]
    assert rate > 0.8


def test_telemetry_reset_clears_stats():
    obs_seq = [
        _obs(**_b(player_x=1)),
        _obs(**_b(player_x=1)),
        _obs(**_b(player_x=2)),
    ]
    env = PokemonTelemetryWrapper(FakeEnv(obs_seq), feature_index=IDX)
    env.reset()
    env.step(ACT)
    _, _, _, _, info = env.step(ACT)
    env.reset()
    env.step(ACT)
    _, _, _, _, info2 = env.step(ACT)
    assert info2["game_stats"]["total_steps"] == 2
