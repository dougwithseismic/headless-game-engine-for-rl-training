"""Pokemon Red game profile for the bridge framework.

Creates a GameBridge that trains an RL agent to play Pokemon Red
via PyBoy. RAM addresses are from the well-documented Pokemon Red
disassembly (pokered).

Observation: RAM features (player position, HP, badges, etc.) +
optional screen pixels.

Actions: MultiBinary(8) — [up, down, left, right, a, b, start, select]

Reward: configurable — default rewards map exploration (new tiles visited).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import gymnasium as gym

from bridges.core.bridge import GameBridge, GameBridgeConfig
from bridges.core.timing import TimingConfig, TimingPolicy
from bridges.emulators.pyboy_host import PyBoyHost
from bridges.sinks.pyboy_sink import PyBoyActionSink
from bridges.sources.pyboy_source import PyBoyObservationSource, RAMFeature
from bridges.resets.pyboy_reset import PyBoyReset


# ---------------------------------------------------------------------------
# Pokemon Red RAM addresses (from pokered disassembly)
# https://github.com/pret/pokered
# ---------------------------------------------------------------------------

POKEMON_RED_RAM = [
    RAMFeature("player_x",          0xD362, 255.0),
    RAMFeature("player_y",          0xD361, 255.0),
    RAMFeature("map_id",            0xD35E, 255.0),
    RAMFeature("player_direction",  0xC109, 255.0),
    RAMFeature("badges",            0xD356, 8.0),    # bit flags, 8 badges
    RAMFeature("party_size",        0xD163, 6.0),    # max 6 pokemon
    RAMFeature("party_hp_1",        0xD16C, 255.0),  # current HP byte 1 of first pokemon
    RAMFeature("party_hp_2",        0xD16D, 255.0),  # current HP byte 2
    RAMFeature("party_max_hp_1",    0xD18D, 255.0),
    RAMFeature("party_max_hp_2",    0xD18E, 255.0),
    RAMFeature("party_level",       0xD18C, 100.0),  # level of first pokemon
    RAMFeature("battle_type",       0xD057, 255.0),  # 0=none, 1=wild, 2=trainer
    RAMFeature("menu_state",        0xD125, 255.0),
    RAMFeature("text_progress",     0xC6AC, 255.0),
    RAMFeature("money_1",           0xD347, 255.0),  # BCD encoded
    RAMFeature("money_2",           0xD348, 255.0),
]


def make_pokemon_red_bridge(
    rom_path: str | Path,
    include_screen: bool = False,
    screen_downscale: int = 4,
    ticks_per_step: int = 24,
    headless: bool = True,
    speed: int = 0,
    save_state_path: str | Path | None = None,
) -> GameBridge:
    """Create a GameBridge for Pokemon Red.

    Args:
        rom_path: Path to Pokemon Red ROM file.
        include_screen: If True, append downscaled screen pixels to observations.
        screen_downscale: Factor to reduce screen resolution (4 = 40x36 grayscale).
        ticks_per_step: Emulator frames per agent step. 24 frames ≈ 0.4s game time.
            Higher values let button presses take effect (menus, movement animations).
        headless: Run without a display window.
        speed: Emulation speed (0 = uncapped, 1 = normal, 2+ = fast-forward).
        save_state_path: Path to a .state file to load as initial checkpoint.
            If None, the initial game state (after boot) is used.
    """
    host = PyBoyHost(
        rom_path=rom_path,
        headless=headless,
        speed=speed,
    )

    sink = PyBoyActionSink(host=host)
    source = PyBoyObservationSource(
        host=host,
        ram_features=POKEMON_RED_RAM,
        include_screen=include_screen,
        screen_downscale=screen_downscale,
        ticks_per_step=ticks_per_step,
        render=include_screen or not headless,
    )
    reset = PyBoyReset(host=host)

    bridge = GameBridge(
        action_sink=sink,
        observation_source=source,
        reset_strategy=reset,
        config=GameBridgeConfig(
            name="pokemon_red",
            timing=TimingConfig(policy=TimingPolicy.FREE_RUNNING),
        ),
    )

    if save_state_path is not None:
        host.start()
        host.load_state_from_file(save_state_path)
        host.save_state_to_buffer("default")

    return bridge


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


class ExplorationReward:
    """Rewards visiting new map tiles. Tracks (map_id, x, y) tuples."""

    def __init__(self, new_tile_reward: float = 1.0, revisit_reward: float = 0.0):
        self.new_tile_reward = new_tile_reward
        self.revisit_reward = revisit_reward
        self.visited: set[tuple[int, int, int]] = set()

    def __call__(self, prev_obs: np.ndarray, action: np.ndarray, obs: np.ndarray) -> float:
        # RAM feature indices match POKEMON_RED_RAM order
        x = int(obs[0] * 255)
        y = int(obs[1] * 255)
        map_id = int(obs[2] * 255)
        tile = (map_id, x, y)

        if tile not in self.visited:
            self.visited.add(tile)
            return self.new_tile_reward
        return self.revisit_reward

    def reset(self) -> None:
        self.visited.clear()


class ProgressReward:
    """Composite reward combining exploration, badges, and HP management."""

    def __init__(self):
        self.exploration = ExplorationReward(new_tile_reward=0.5)
        self._prev_badges = 0

    def __call__(self, prev_obs: np.ndarray, action: np.ndarray, obs: np.ndarray) -> float:
        reward = self.exploration(prev_obs, action, obs)

        # Badge reward (big bonus for new badges)
        badges = int(obs[4] * 8)
        if badges > self._prev_badges:
            reward += 10.0 * (badges - self._prev_badges)
            self._prev_badges = badges

        # HP penalty (encourage keeping pokemon healthy)
        hp_frac = obs[6] / max(obs[8], 0.01)
        if hp_frac < 0.2:
            reward -= 0.1

        return reward

    def reset(self) -> None:
        self.exploration.reset()
        self._prev_badges = 0
