"""Tests for the Pokemon Gold profile — no ROM needed."""

import numpy as np
import pytest

from bridges.profiles.pokemon_gold import (
    POKEMON_GOLD_RAM,
    FEATURE_GROUPS,
    decode_bcd_money,
    decode_hp,
    count_badges,
    raw_byte,
    ExplorationReward,
    ProgressReward,
    BattleReward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_decode_bcd_money():
    assert decode_bcd_money(0x00, 0x00, 0x00) == 0
    assert decode_bcd_money(0x00, 0x01, 0x23) == 123
    assert decode_bcd_money(0x12, 0x34, 0x56) == 123456
    assert decode_bcd_money(0x99, 0x99, 0x99) == 999999


def test_decode_hp():
    assert decode_hp(0, 0) == 0
    assert decode_hp(0, 50) == 50
    assert decode_hp(1, 0) == 256
    assert decode_hp(0xFF, 0xFF) == 65535


def test_count_badges():
    assert count_badges(0) == 0
    assert count_badges(0b00000001) == 1
    assert count_badges(0b10101010) == 4
    assert count_badges(0xFF) == 8


def test_raw_byte():
    idx = {"player_x": 0, "player_y": 1}
    obs = np.array([100 / 255, 200 / 255], dtype=np.float32)
    assert raw_byte(obs, idx, "player_x", 255) == 100
    assert raw_byte(obs, idx, "player_y", 255) == 200


# ---------------------------------------------------------------------------
# RAM address sanity
# ---------------------------------------------------------------------------


def test_ram_addresses_unique():
    addrs = [f.address for f in POKEMON_GOLD_RAM]
    assert len(addrs) == len(set(addrs)), "Duplicate RAM addresses"


def test_ram_names_unique():
    names = [f.name for f in POKEMON_GOLD_RAM]
    assert len(names) == len(set(names)), "Duplicate feature names"


def test_ram_addresses_in_wram():
    for f in POKEMON_GOLD_RAM:
        assert 0xC000 <= f.address <= 0xDFFF, f"{f.name} address 0x{f.address:04X} outside WRAM"


def test_feature_groups_cover_all_ram():
    total = sum(g.length for g in FEATURE_GROUPS)
    assert total == len(POKEMON_GOLD_RAM)


def test_feature_groups_no_overlap():
    ranges = [(g.start, g.start + g.length) for g in FEATURE_GROUPS]
    ranges.sort()
    for i in range(len(ranges) - 1):
        assert ranges[i][1] <= ranges[i + 1][0], f"Overlap: {FEATURE_GROUPS[i].name} and {FEATURE_GROUPS[i+1].name}"


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def _make_obs(idx: dict[str, int], overrides: dict[str, float]) -> np.ndarray:
    """Build a fake obs array with specific values at named indices."""
    obs = np.zeros(len(POKEMON_GOLD_RAM), dtype=np.float32)
    for name, val in overrides.items():
        if name in idx:
            obs[idx[name]] = val
    return obs


def test_exploration_reward_new_tile():
    names = [f.name for f in POKEMON_GOLD_RAM]
    idx = {n: i for i, n in enumerate(names)}
    reward_fn = ExplorationReward(idx)

    obs = _make_obs(idx, {"player_x": 5 / 255, "player_y": 10 / 255, "map_group": 1 / 255, "map_number": 2 / 255})
    prev = np.zeros_like(obs)
    r = reward_fn(prev, np.zeros(8), obs)
    assert r == 2.0  # first visit: 2.0 / sqrt(1)

    # Same tile, no movement → 0
    r2 = reward_fn(obs, np.zeros(8), obs)
    assert r2 == 0.0

    # Visit same tile again from a different position → diminished reward
    prev2 = _make_obs(idx, {"player_x": 4 / 255, "player_y": 10 / 255, "map_group": 1 / 255, "map_number": 2 / 255})
    r3 = reward_fn(prev2, np.zeros(8), obs)
    assert 1.0 < r3 < 2.0  # 2.0 / sqrt(2) ≈ 1.41


def test_exploration_reward_no_bounce():
    """Back-and-forth between 2 tiles should not reward."""
    names = [f.name for f in POKEMON_GOLD_RAM]
    idx = {n: i for i, n in enumerate(names)}
    reward_fn = ExplorationReward(idx)

    a = _make_obs(idx, {"player_x": 5 / 255, "player_y": 5 / 255})
    b = _make_obs(idx, {"player_x": 6 / 255, "player_y": 5 / 255})

    reward_fn(a, np.zeros(8), b)  # a→b: first visit to b
    r = reward_fn(b, np.zeros(8), a)  # b→a: first visit to a
    # a→b again: bounce back to b (prev_tile is a)
    r_bounce = reward_fn(a, np.zeros(8), b)
    # This is a revisit to b from a, count=2 → 2/sqrt(2)
    assert r_bounce < 2.0


def test_exploration_reward_reset():
    names = [f.name for f in POKEMON_GOLD_RAM]
    idx = {n: i for i, n in enumerate(names)}
    reward_fn = ExplorationReward(idx)

    obs = _make_obs(idx, {"player_x": 1 / 255, "player_y": 1 / 255})
    reward_fn(np.zeros_like(obs), np.zeros(8), obs)
    reward_fn.reset()
    r = reward_fn(np.zeros_like(obs), np.zeros(8), obs)
    assert r == 2.0  # fresh counts after reset


def test_progress_reward_badge_bonus():
    names = [f.name for f in POKEMON_GOLD_RAM]
    idx = {n: i for i, n in enumerate(names)}
    reward_fn = ProgressReward(idx)

    obs0 = _make_obs(idx, {"johto_badges": 0, "party1_level": 5 / 100})
    obs1 = _make_obs(idx, {"johto_badges": 1 / 255, "party1_level": 5 / 100})
    r = reward_fn(obs0, np.zeros(8), obs1)
    assert r > 10  # badge bonus of 20


def test_progress_reward_level_bonus():
    names = [f.name for f in POKEMON_GOLD_RAM]
    idx = {n: i for i, n in enumerate(names)}
    reward_fn = ProgressReward(idx)

    obs0 = _make_obs(idx, {"party1_level": 5 / 100})
    reward_fn(np.zeros_like(obs0), np.zeros(8), obs0)

    obs1 = _make_obs(idx, {
        "party1_level": 7 / 100,
        "player_x": 1 / 255,
    })
    r = reward_fn(obs0, np.zeros(8), obs1)
    assert r > 3.0  # 2.0 * 2 level ups + exploration


def test_battle_reward_win():
    names = [f.name for f in POKEMON_GOLD_RAM]
    idx = {n: i for i, n in enumerate(names)}
    reward_fn = BattleReward(idx)

    # Enter battle
    obs_battle = _make_obs(idx, {
        "battle_mode": 1 / 2,
        "party1_hp_hi": 0, "party1_hp_lo": 50 / 255,
        "party1_maxhp_hi": 0, "party1_maxhp_lo": 50 / 255,
    })
    reward_fn(np.zeros_like(obs_battle), np.zeros(8), obs_battle)

    # Win battle (back to overworld with HP remaining)
    obs_won = _make_obs(idx, {
        "battle_mode": 0,
        "party1_hp_hi": 0, "party1_hp_lo": 40 / 255,
        "party1_maxhp_hi": 0, "party1_maxhp_lo": 50 / 255,
    })
    r = reward_fn(obs_battle, np.zeros(8), obs_won)
    assert r > 5.0  # win bonus + hp preserved bonus


def test_battle_reward_faint():
    names = [f.name for f in POKEMON_GOLD_RAM]
    idx = {n: i for i, n in enumerate(names)}
    reward_fn = BattleReward(idx)

    obs_battle = _make_obs(idx, {
        "battle_mode": 1 / 2,
        "party1_hp_hi": 0, "party1_hp_lo": 50 / 255,
        "party1_maxhp_hi": 0, "party1_maxhp_lo": 50 / 255,
    })
    reward_fn(np.zeros_like(obs_battle), np.zeros(8), obs_battle)

    obs_faint = _make_obs(idx, {
        "battle_mode": 0,
        "party1_hp_hi": 0, "party1_hp_lo": 0,
        "party1_maxhp_hi": 0, "party1_maxhp_lo": 50 / 255,
    })
    r = reward_fn(obs_battle, np.zeros(8), obs_faint)
    assert r < 0  # faint penalty
