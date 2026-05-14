"""Milestone reward: large bonuses for reaching specific maps/locations.

Combines map-based milestones with count-based exploration. The agent
gets big rewards for game progress (reaching Route 29, Cherrygrove, etc.)
and small rewards for exploring new tiles along the way.

Milestones fire once per episode via high-water-mark tracking.
Exploration uses 1/sqrt(visit_count) diminishing returns.

Usage:
    reward_fn = MilestoneReward(feature_index)
    env = ExternalGameGym(bridge=bridge, reward_fn=reward_fn)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Known map (group, number) values for Pokemon Gold
MAPS = {
    "new_bark_town": (24, 4),
    "elm_lab": (24, 5),
    "player_house_1f": (24, 6),
    "player_house_2f": (24, 7),
    "route_29": (24, 3),
    # These need to be discovered by manual play:
    # "cherrygrove": (TBD, TBD),
    # "route_30": (TBD, TBD),
    # "mr_pokemon_house": (TBD, TBD),
}


@dataclass
class Milestone:
    name: str
    map_group: int
    map_number: int
    reward: float


# Ordered by expected progression
DEFAULT_MILESTONES = [
    Milestone("reach_route_29", 24, 3, 50.0),
    # Add more as map values are discovered:
    # Milestone("reach_cherrygrove", TBD, TBD, 100.0),
    # Milestone("reach_route_30", TBD, TBD, 150.0),
]


def _raw(obs: np.ndarray, idx: dict[str, int], name: str, norm: float) -> int:
    if name not in idx:
        return 0
    return int(round(obs[idx[name]] * norm))


class MilestoneReward:
    """Map milestone rewards + count-based exploration + dialogue bonus.

    Rewards:
    - Milestone: +50/+100/+150 for reaching specific maps (once per episode)
    - Exploration: 1.0/sqrt(N) for tile visits (diminishing returns)
    - Movement: always some reward for changing position
    - Dialogue: +0.1 for pressing A when textbox is active
    """

    def __init__(
        self,
        feature_index: dict[str, int],
        milestones: list[Milestone] | None = None,
        explore_bonus: float = 1.0,
        dialogue_bonus: float = 0.1,
    ):
        self.idx = feature_index
        self.milestones = milestones or DEFAULT_MILESTONES
        self.explore_bonus = explore_bonus
        self.dialogue_bonus = dialogue_bonus

        self._visit_counts: dict[tuple, int] = {}
        self._prev_tile: tuple | None = None
        self._achieved: set[str] = set()
        self._hwm_level = 0
        self._hwm_badges = 0
        self._hwm_party = 0

    def __call__(self, prev_obs: np.ndarray, action: np.ndarray, obs: np.ndarray) -> float:
        reward = 0.0
        idx = self.idx

        mg = _raw(obs, idx, "map_group", 255)
        mn = _raw(obs, idx, "map_number", 255)
        x = _raw(obs, idx, "player_x", 255)
        y = _raw(obs, idx, "player_y", 255)

        # Skip glitch frames
        if mg == 0 and mn == 0:
            return 0.0

        tile = (mg, mn, x, y)
        prev_x = _raw(prev_obs, idx, "player_x", 255)
        prev_y = _raw(prev_obs, idx, "player_y", 255)
        moved = (x != prev_x or y != prev_y)

        # Milestone rewards (big, once per episode)
        for ms in self.milestones:
            if ms.name not in self._achieved and mg == ms.map_group and mn == ms.map_number:
                self._achieved.add(ms.name)
                reward += ms.reward

        # Count-based exploration (diminishing returns)
        if moved and tile != self._prev_tile:
            self._prev_tile = (mg, mn, prev_x, prev_y)
            count = self._visit_counts.get(tile, 0) + 1
            self._visit_counts[tile] = count
            reward += self.explore_bonus / (count ** 0.5)

        # Dialogue bonus: reward pressing A when textbox is active
        prev_text = _raw(prev_obs, idx, "textbox_flags", 255)
        curr_text = _raw(obs, idx, "textbox_flags", 255)
        if prev_text > 0 and curr_text != prev_text:
            reward += self.dialogue_bonus

        # Level up (high-water-mark)
        level = _raw(obs, idx, "party1_level", 100)
        if level > self._hwm_level:
            reward += 3.0 * (level - self._hwm_level)
            self._hwm_level = level

        # Badge (high-water-mark)
        johto = bin(_raw(obs, idx, "johto_badges", 255)).count("1")
        kanto = bin(_raw(obs, idx, "kanto_badges", 255)).count("1")
        total = johto + kanto
        if total > self._hwm_badges:
            reward += 20.0 * (total - self._hwm_badges)
            self._hwm_badges = total

        # Party size (high-water-mark)
        party = _raw(obs, idx, "party_size", 6)
        if party > self._hwm_party:
            reward += 5.0
            self._hwm_party = party

        return reward

    def reset(self) -> None:
        self._visit_counts.clear()
        self._prev_tile = None
        self._achieved.clear()
        self._hwm_level = 0
        self._hwm_badges = 0
        self._hwm_party = 0
