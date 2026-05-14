"""Pokemon game telemetry: detect and log game events from observation deltas.

Sits as a wrapper around the gym env. Compares consecutive observations to
detect map transitions, battles, level ups, badge milestones, etc. Emits
events as structured dicts and tracks cumulative stats for TensorBoard.

Uses high-water-mark tracking for level/badges/party to avoid false positives
from momentary RAM glitches (GBC WRAM bank switching can cause brief 0-reads).

Usage:
    env = ExternalGameGym(bridge=bridge, reward_fn=reward_fn)
    env = PokemonTelemetryWrapper(env, feature_index=bridge.feature_index)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np


@dataclass
class GameEvent:
    step: int
    event_type: str
    data: dict


@dataclass
class GameStats:
    tiles_visited: int = 0
    maps_visited: int = 0
    battles_won: int = 0
    battles_lost: int = 0
    wild_battles: int = 0
    trainer_battles: int = 0
    level_ups: int = 0
    max_level: int = 0
    badges_earned: int = 0
    party_size_max: int = 0
    steps_in_battle: int = 0
    steps_in_menu: int = 0
    steps_stuck: int = 0
    total_steps: int = 0
    maps_seen: set = field(default_factory=set)
    tiles_seen: set = field(default_factory=set)

    def as_dict(self) -> dict:
        return {
            "tiles_visited": self.tiles_visited,
            "maps_visited": self.maps_visited,
            "battles_won": self.battles_won,
            "battles_lost": self.battles_lost,
            "wild_battles": self.wild_battles,
            "trainer_battles": self.trainer_battles,
            "level_ups": self.level_ups,
            "max_level": self.max_level,
            "badges_earned": self.badges_earned,
            "party_size_max": self.party_size_max,
            "steps_in_battle": self.steps_in_battle,
            "steps_in_menu": self.steps_in_menu,
            "steps_stuck": self.steps_stuck,
            "total_steps": self.total_steps,
            "exploration_rate": self.tiles_visited / max(self.total_steps, 1),
        }


def _raw(obs: np.ndarray, idx: dict[str, int], name: str, norm: float) -> int:
    if name not in idx:
        return 0
    return int(round(obs[idx[name]] * norm))


def _count_bits(val: int) -> int:
    return bin(val).count("1")


class PokemonTelemetryWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, feature_index: dict[str, int] | None = None):
        super().__init__(env)
        self.idx = feature_index or getattr(env, "feature_index", {})
        self._stats = GameStats()
        self._events: list[GameEvent] = []
        self._prev_obs: np.ndarray | None = None
        self._step_count = 0
        self._prev_pos: tuple | None = None
        self._stuck_count = 0
        # High-water marks to filter RAM glitches
        self._hwm_level = 0
        self._hwm_badges_johto = 0
        self._hwm_badges_kanto = 0
        self._hwm_party_size = 0
        self._in_battle = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._stats = GameStats()
        self._events = []
        self._prev_obs = obs.copy()
        self._step_count = 0
        self._prev_pos = self._pos(obs)
        self._stuck_count = 0

        level = _raw(obs, self.idx, "party1_level", 100)
        party = _raw(obs, self.idx, "party_size", 6)
        johto = _count_bits(_raw(obs, self.idx, "johto_badges", 255))
        kanto = _count_bits(_raw(obs, self.idx, "kanto_badges", 255))

        self._hwm_level = level
        self._hwm_badges_johto = johto
        self._hwm_badges_kanto = kanto
        self._hwm_party_size = party
        self._in_battle = _raw(obs, self.idx, "battle_mode", 2) > 0

        self._stats.max_level = level
        self._stats.party_size_max = party
        self._stats.badges_earned = johto + kanto
        self._track_tile(obs)

        init_map = (_raw(obs, self.idx, "map_group", 255), _raw(obs, self.idx, "map_number", 255))
        if init_map != (0, 0):
            self._stats.maps_seen.add(init_map)
            self._stats.maps_visited = 1

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._stats.total_steps = self._step_count

        self._track_tile(obs)
        self._detect_events(obs)
        self._prev_obs = obs.copy()

        if terminated or truncated:
            info["game_stats"] = self._stats.as_dict()
            info["game_events"] = [
                {"step": e.step, "type": e.event_type, "data": e.data}
                for e in self._events[-20:]
            ]

        return obs, reward, terminated, truncated, info

    def _detect_events(self, obs: np.ndarray):
        idx = self.idx

        # Position tracking
        pos = self._pos(obs)
        if pos == self._prev_pos:
            self._stuck_count += 1
            if self._stuck_count > 10:
                self._stats.steps_stuck += 1
        else:
            self._stuck_count = 0
        self._prev_pos = pos

        # Map transition — ignore transitions to/from (0,0) which are RAM glitches
        if self._prev_obs is not None:
            prev_map = (_raw(self._prev_obs, idx, "map_group", 255), _raw(self._prev_obs, idx, "map_number", 255))
            curr_map = (_raw(obs, idx, "map_group", 255), _raw(obs, idx, "map_number", 255))
            if curr_map != prev_map and curr_map != (0, 0) and prev_map != (0, 0):
                is_new = curr_map not in self._stats.maps_seen
                self._stats.maps_seen.add(curr_map)
                self._stats.maps_visited = len(self._stats.maps_seen)
                if is_new:
                    self._emit("map_enter", {"map_group": curr_map[0], "map_number": curr_map[1], "new": True})

        # Battle transitions — use state machine, ignore glitch reads
        curr_battle = _raw(obs, idx, "battle_mode", 2)
        if curr_battle > 0:
            self._stats.steps_in_battle += 1
            if not self._in_battle:
                self._in_battle = True
                if curr_battle == 1:
                    self._stats.wild_battles += 1
                    self._emit("battle_start", {"type": "wild"})
                elif curr_battle == 2:
                    self._stats.trainer_battles += 1
                    self._emit("battle_start", {"type": "trainer"})
        elif curr_battle == 0 and self._in_battle:
            hp = _raw(obs, idx, "party1_hp_hi", 255) * 256 + _raw(obs, idx, "party1_hp_lo", 255)
            if hp > 0:
                self._stats.battles_won += 1
                self._emit("battle_end", {"result": "won", "hp_remaining": hp})
            else:
                self._stats.battles_lost += 1
                self._emit("battle_end", {"result": "lost"})
            self._in_battle = False

        # Level up — high-water mark: only count when level EXCEEDS previous max
        level = _raw(obs, idx, "party1_level", 100)
        if level > self._hwm_level and level > 0:
            gained = level - self._hwm_level
            self._stats.level_ups += gained
            self._stats.max_level = level
            self._emit("level_up", {"from": self._hwm_level, "to": level})
            self._hwm_level = level

        # Badge earned — high-water mark
        johto = _count_bits(_raw(obs, idx, "johto_badges", 255))
        kanto = _count_bits(_raw(obs, idx, "kanto_badges", 255))
        if johto > self._hwm_badges_johto:
            new_badges = johto - self._hwm_badges_johto
            self._stats.badges_earned += new_badges
            self._emit("badge", {"region": "johto", "count": johto})
            self._hwm_badges_johto = johto
        if kanto > self._hwm_badges_kanto:
            new_badges = kanto - self._hwm_badges_kanto
            self._stats.badges_earned += new_badges
            self._emit("badge", {"region": "kanto", "count": kanto})
            self._hwm_badges_kanto = kanto

        # Party size — high-water mark
        party = _raw(obs, idx, "party_size", 6)
        if party > self._hwm_party_size:
            self._stats.party_size_max = party
            self._emit("party_change", {"from": self._hwm_party_size, "to": party})
            self._hwm_party_size = party

        # Menu time
        if _raw(obs, idx, "textbox_flags", 255) > 0:
            self._stats.steps_in_menu += 1

    def _track_tile(self, obs: np.ndarray):
        tile = (
            _raw(obs, self.idx, "map_group", 255),
            _raw(obs, self.idx, "map_number", 255),
            _raw(obs, self.idx, "player_x", 255),
            _raw(obs, self.idx, "player_y", 255),
        )
        if tile[0] == 0 and tile[1] == 0:
            return
        if tile not in self._stats.tiles_seen:
            self._stats.tiles_seen.add(tile)
            self._stats.tiles_visited = len(self._stats.tiles_seen)

    def _pos(self, obs: np.ndarray) -> tuple:
        return (
            _raw(obs, self.idx, "map_group", 255),
            _raw(obs, self.idx, "map_number", 255),
            _raw(obs, self.idx, "player_x", 255),
            _raw(obs, self.idx, "player_y", 255),
        )

    def _emit(self, event_type: str, data: dict):
        self._events.append(GameEvent(self._step_count, event_type, data))
