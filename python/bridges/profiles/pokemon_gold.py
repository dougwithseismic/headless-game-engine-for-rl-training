"""Pokemon Gold (GBC) game profile for the bridge framework.

RAM addresses sourced from the pret/pokegold disassembly via
pokegold.sym (pokemon-speedrunning/symfiles). Cross-referenced
against DataCrystal Gold/Silver RAM map.

NOTE: Gold's WRAM layout is completely different from Red/Blue.
Party starts at 0xDA22 (Red: 0xD163), coords at 0xDA02-DA03
(Red: 0xD361-D362), and maps use group+number (Red: single ID).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import gymnasium as gym

from bridges.core.bridge import GameBridge, GameBridgeConfig
from bridges.core.timing import TimingConfig, TimingPolicy
from bridges.core.obs_source import FeatureGroup
from bridges.emulators.pyboy_host import PyBoyHost
from bridges.sinks.pyboy_sink import PyBoyActionSink
from bridges.sources.pyboy_source import PyBoyObservationSource, RAMFeature
from bridges.resets.pyboy_reset import PyBoyReset


# ---------------------------------------------------------------------------
# Pokemon Gold RAM addresses
# Source: pret/pokegold → pokegold.sym
# https://github.com/pokemon-speedrunning/symfiles/blob/master/pokegold.sym
# ---------------------------------------------------------------------------

POKEMON_GOLD_RAM = [
    # Player position
    RAMFeature("player_x",          0xDA03, 255.0),   # wXCoord
    RAMFeature("player_y",          0xDA02, 255.0),   # wYCoord
    RAMFeature("map_group",         0xDA00, 255.0),   # wMapGroup
    RAMFeature("map_number",        0xDA01, 255.0),   # wMapNumber
    RAMFeature("player_direction",  0xD205, 12.0),    # wPlayerDirection: 0/4/8/12
    RAMFeature("facing_direction",  0xCF2F, 3.0),     # wFacingDirection: 0-3

    # Badges (bitfields)
    RAMFeature("johto_badges",      0xD57C, 255.0),   # wJohtoBadges
    RAMFeature("kanto_badges",      0xD57D, 255.0),   # wKantoBadges

    # Party
    RAMFeature("party_size",        0xDA22, 6.0),     # wPartyCount
    RAMFeature("party1_hp_hi",      0xDA4C, 255.0),   # wPartyMon1HP (big-endian)
    RAMFeature("party1_hp_lo",      0xDA4D, 255.0),
    RAMFeature("party1_maxhp_hi",   0xDA4E, 255.0),   # wPartyMon1MaxHP
    RAMFeature("party1_maxhp_lo",   0xDA4F, 255.0),
    RAMFeature("party1_level",      0xDA49, 100.0),   # wPartyMon1Level

    # Battle
    RAMFeature("battle_mode",       0xD116, 2.0),     # wBattleMode: 0=none, 1=wild, 2=trainer
    RAMFeature("battle_type",       0xD119, 255.0),   # wBattleType

    # Money (3 bytes BCD)
    RAMFeature("money_hi",          0xD573, 255.0),   # wMoney byte 0
    RAMFeature("money_mid",         0xD574, 255.0),   # wMoney byte 1
    RAMFeature("money_lo",          0xD575, 255.0),   # wMoney byte 2

    # Step counter & movement
    RAMFeature("step_count",        0xD9BD, 255.0),   # wStepCount (wraps at 256)
    RAMFeature("player_state",      0xD682, 255.0),   # wPlayerState (normal/bike/surf)
    RAMFeature("player_action",     0xD208, 255.0),   # wPlayerAction
    RAMFeature("standing_tile",     0xD20B, 255.0),   # wPlayerStandingTile

    # Menu / text state
    RAMFeature("textbox_flags",     0xD19C, 255.0),   # wTextboxFlags
    RAMFeature("menu_flags",        0xCEB8, 255.0),   # wMenuFlags
]

FEATURE_GROUPS = [
    FeatureGroup("position", start=0, length=6),
    FeatureGroup("badges", start=6, length=2),
    FeatureGroup("party", start=8, length=6),
    FeatureGroup("battle", start=14, length=2),
    FeatureGroup("money", start=16, length=3),
    FeatureGroup("movement", start=19, length=4),
    FeatureGroup("menu", start=23, length=2),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_bcd_money(money_hi: int, money_mid: int, money_lo: int) -> int:
    """Decode 3-byte BCD money. Pass raw (un-normalized) byte values."""
    def _bcd(b: int) -> int:
        return (b >> 4) * 10 + (b & 0x0F)
    return _bcd(money_hi) * 10000 + _bcd(money_mid) * 100 + _bcd(money_lo)


def decode_hp(hi: int, lo: int) -> int:
    """Decode 2-byte big-endian HP. Pass raw byte values."""
    return (hi << 8) | lo


def count_badges(badge_byte: int) -> int:
    """Count set bits in a badge bitfield byte."""
    return bin(int(badge_byte)).count("1")


def raw_byte(obs: np.ndarray, idx: dict[str, int], name: str, norm: float) -> int:
    """Recover the raw integer byte from a normalized observation value."""
    return int(round(obs[idx[name]] * norm))


# ---------------------------------------------------------------------------
# Bridge factory
# ---------------------------------------------------------------------------

def make_pokemon_gold_bridge(
    rom_path: str | Path,
    include_screen: bool = False,
    screen_downscale: int = 4,
    ticks_per_step: int = 24,
    headless: bool = True,
    speed: int = 0,
    save_state_path: str | Path | None = None,
) -> GameBridge:
    """Create a GameBridge for Pokemon Gold.

    Args:
        rom_path: Path to Pokemon Gold ROM (.gbc).
        include_screen: Append downscaled screen pixels to observations.
        screen_downscale: Factor to reduce screen (4 = 40x36 grayscale).
        ticks_per_step: Emulator frames per agent step. 24 ≈ 0.4s game time.
        headless: Run without display window.
        speed: 0 = uncapped, 1 = normal, 2+ = fast-forward.
        save_state_path: .state file to load as initial checkpoint.
    """
    host = PyBoyHost(
        rom_path=rom_path,
        headless=headless,
        speed=speed,
        cgb=True,
    )

    sink = PyBoyActionSink(host=host, buttons=("up", "down", "left", "right", "a", "b"))
    source = PyBoyObservationSource(
        host=host,
        ram_features=POKEMON_GOLD_RAM,
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
            name="pokemon_gold",
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
    """Count-based exploration: reward decays with sqrt(visit_count).

    First visit: +new_tile_reward (2.0)
    Second visit: +2.0/sqrt(2) = 1.41
    Third visit: +2.0/sqrt(3) = 1.15
    ...
    Nth visit: +2.0/sqrt(N)

    This ensures there's always SOME reward for moving to less-visited
    tiles, while heavily-visited areas give diminishing returns. Prevents
    both "stand still" (no reward) and "oscillate between 2 tiles" (rapidly
    diminishing) failure modes.
    """

    def __init__(
        self,
        feature_index: dict[str, int],
        new_tile_reward: float = 2.0,
    ):
        self.idx = feature_index
        self.new_tile_reward = new_tile_reward
        self._visit_counts: dict[tuple[int, int, int, int], int] = {}
        self._prev_tile: tuple[int, int, int, int] | None = None

    def __call__(self, prev_obs: np.ndarray, action: np.ndarray, obs: np.ndarray) -> float:
        x = raw_byte(obs, self.idx, "player_x", 255)
        y = raw_byte(obs, self.idx, "player_y", 255)
        mg = raw_byte(obs, self.idx, "map_group", 255)
        mn = raw_byte(obs, self.idx, "map_number", 255)
        tile = (mg, mn, x, y)

        prev_x = raw_byte(prev_obs, self.idx, "player_x", 255)
        prev_y = raw_byte(prev_obs, self.idx, "player_y", 255)
        moved = (x != prev_x or y != prev_y)

        if not moved:
            return 0.0

        # Don't reward bouncing back to where you just were
        if tile == self._prev_tile:
            return 0.0

        self._prev_tile = (mg, mn, prev_x, prev_y)
        count = self._visit_counts.get(tile, 0) + 1
        self._visit_counts[tile] = count

        return self.new_tile_reward / (count ** 0.5)

    def reset(self) -> None:
        self._visit_counts.clear()
        self._prev_tile = None


class ProgressReward:
    """Composite reward: exploration + movement + badges + level.

    Positive-only exploration (no revisit penalty) to avoid the agent
    learning to stand still. Movement reward ensures activity.

    Usage:
        bridge = make_pokemon_gold_bridge(...)
        reward = ProgressReward(bridge.feature_index)
        env = ExternalGameGym(bridge=bridge, reward_fn=reward)
    """

    def __init__(self, feature_index: dict[str, int]):
        self.idx = feature_index
        self.exploration = ExplorationReward(feature_index, new_tile_reward=1.0)
        self._hwm_badges = 0
        self._hwm_level = 0
        self._hwm_party_size = 0

    def __call__(self, prev_obs: np.ndarray, action: np.ndarray, obs: np.ndarray) -> float:
        reward = self.exploration(prev_obs, action, obs)

        # Badge milestone (high-water mark to avoid RAM glitch double-counting)
        johto = count_badges(raw_byte(obs, self.idx, "johto_badges", 255))
        kanto = count_badges(raw_byte(obs, self.idx, "kanto_badges", 255))
        total_badges = johto + kanto
        if total_badges > self._hwm_badges:
            reward += 20.0 * (total_badges - self._hwm_badges)
            self._hwm_badges = total_badges

        # Level up (high-water mark)
        level = raw_byte(obs, self.idx, "party1_level", 100)
        if level > self._hwm_level:
            reward += 3.0 * (level - self._hwm_level)
            self._hwm_level = level

        # New party member (high-water mark)
        party = raw_byte(obs, self.idx, "party_size", 6)
        if party > self._hwm_party_size:
            reward += 5.0
            self._hwm_party_size = party

        return reward

    def reset(self) -> None:
        self.exploration.reset()
        self._hwm_badges = 0
        self._hwm_level = 0
        self._hwm_party_size = 0


class BattleReward:
    """Focused reward for battle training: win fights, don't faint."""

    def __init__(self, feature_index: dict[str, int]):
        self.idx = feature_index
        self._was_in_battle = False
        self._battle_start_hp = 0

    def __call__(self, prev_obs: np.ndarray, action: np.ndarray, obs: np.ndarray) -> float:
        reward = 0.0
        battle = raw_byte(obs, self.idx, "battle_mode", 2)
        hp = decode_hp(
            raw_byte(obs, self.idx, "party1_hp_hi", 255),
            raw_byte(obs, self.idx, "party1_hp_lo", 255),
        )
        max_hp = decode_hp(
            raw_byte(obs, self.idx, "party1_maxhp_hi", 255),
            raw_byte(obs, self.idx, "party1_maxhp_lo", 255),
        )

        if battle > 0 and not self._was_in_battle:
            self._battle_start_hp = hp
            self._was_in_battle = True

        if battle == 0 and self._was_in_battle:
            if hp > 0:
                reward += 5.0
                if max_hp > 0:
                    hp_preserved = hp / max_hp
                    reward += 2.0 * hp_preserved
            else:
                reward -= 5.0
            self._was_in_battle = False

        if battle > 0 and max_hp > 0:
            hp_frac = hp / max_hp
            if hp_frac < 0.1:
                reward -= 0.5

        return reward

    def reset(self) -> None:
        self._was_in_battle = False
        self._battle_start_hp = 0
