"""Live demo: Pokemon Gold running through the GhostLobby bridge framework.

Shows the game in a window with real RAM-based observations.
Play manually or watch the scripted sequence.

Usage:
    python demo_gold.py
    python demo_gold.py --state data/pokemon_gold_ready.state
"""

import argparse
import os

import numpy as np

from bridges.profiles.pokemon_gold import make_pokemon_gold_bridge, decode_hp, count_badges, raw_byte
from glgym.gym_external import ExternalGameGym

ROM = os.environ.get("POKEMON_ROM", "data/pokemon_gold.gbc")

# Button arrays: [up, down, left, right, a, b, start, select]
IDLE = np.array([0, 0, 0, 0, 0, 0, 0, 0])
A = np.array([0, 0, 0, 0, 1, 0, 0, 0])
B = np.array([0, 0, 0, 0, 0, 1, 0, 0])
START = np.array([0, 0, 0, 0, 0, 0, 1, 0])
RIGHT = np.array([0, 0, 0, 1, 0, 0, 0, 0])
DOWN = np.array([0, 1, 0, 0, 0, 0, 0, 0])
LEFT = np.array([0, 0, 1, 0, 0, 0, 0, 0])
UP = np.array([1, 0, 0, 0, 0, 0, 0, 0])


def run(rom_path: str, state_path: str | None):
    bridge = make_pokemon_gold_bridge(
        rom_path=rom_path,
        headless=False,
        speed=1,
        ticks_per_step=12,
        save_state_path=state_path,
    )
    idx = bridge.feature_index

    env = ExternalGameGym(bridge=bridge, max_steps=600)
    obs, _ = env.reset()

    # Build a scripted sequence
    seq = []
    if state_path:
        # Already past menus, just walk around
        seq += [DOWN] * 30 + [RIGHT] * 30 + [UP] * 30 + [LEFT] * 30
        seq += [A] * 5 + [IDLE] * 5
        seq += [RIGHT] * 20 + [DOWN] * 20 + [LEFT] * 20 + [UP] * 20
    else:
        # Boot: mash through title and menus
        seq += [IDLE] * 30
        for _ in range(8):
            seq += [A] * 5 + [IDLE] * 15
        seq += [START] * 5 + [IDLE] * 15
        for _ in range(4):
            seq += [A] * 5 + [IDLE] * 15
        seq += [DOWN] * 5 + [A] * 5 + [IDLE] * 15
        for _ in range(3):
            seq += [A] * 5 + [IDLE] * 15
        # Walk around
        seq += [DOWN] * 30 + [RIGHT] * 30 + [UP] * 30 + [LEFT] * 30
        seq += [DOWN] * 20 + [A] * 5 + [IDLE] * 5
        seq += [RIGHT] * 30 + [UP] * 20
        seq += [LEFT] * 30 + [DOWN] * 20

    total_steps = min(len(seq), 500)
    print(f"Pokemon Gold — {total_steps} steps, watch the PyBoy window!")

    for step in range(total_steps):
        obs, r, t, tr, info = env.step(seq[step])

        if step % 50 == 0:
            x = raw_byte(obs, idx, "player_x", 255) if "player_x" in idx else "?"
            y = raw_byte(obs, idx, "player_y", 255) if "player_y" in idx else "?"
            lvl = raw_byte(obs, idx, "party1_level", 100) if "party1_level" in idx else "?"
            badges = count_badges(raw_byte(obs, idx, "johto_badges", 255)) if "johto_badges" in idx else "?"
            print(f"  Step {step:3d}  pos=({x},{y})  lv={lvl}  badges={badges}")

        if t or tr:
            obs, _ = env.reset()

    print("Done!")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", default=ROM)
    parser.add_argument("--state", default=None)
    args = parser.parse_args()
    run(args.rom, args.state)
