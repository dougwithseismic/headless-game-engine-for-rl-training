"""Debug tool: run Pokemon Gold headed with live RAM readout.

Shows real-time values for all tracked RAM addresses in the terminal
while you play in the PyBoy window. Use this to verify that RAM offsets
are correct before training.

Usage:
    python debug_gold_ram.py
    python debug_gold_ram.py --state data/pokemon_gold_ready.state
    python debug_gold_ram.py --speed 2

Controls: play normally in the PyBoy window. Terminal updates every step.
Press Ctrl+C to stop.
"""

import argparse
import os
import sys
import time

from bridges.emulators.pyboy_host import PyBoyHost
from bridges.profiles.pokemon_gold import (
    POKEMON_GOLD_RAM,
    decode_bcd_money,
    decode_hp,
    count_badges,
)

ROM = os.environ.get("POKEMON_ROM", "data/pokemon_gold.gbc")


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def format_direction(val: int) -> str:
    return {0: "down", 4: "up", 8: "left", 12: "right"}.get(val, f"?({val})")


def format_facing(val: int) -> str:
    return {0: "down", 1: "up", 2: "left", 3: "right"}.get(val, f"?({val})")


def format_badges(byte_val: int) -> str:
    names = ["Zephyr", "Hive", "Plain", "Fog", "Mineral", "Storm", "Glacier", "Rising"]
    earned = [n for i, n in enumerate(names) if byte_val & (1 << i)]
    return f"{count_badges(byte_val)}/8 [{', '.join(earned) or 'none'}]"


def format_kanto_badges(byte_val: int) -> str:
    names = ["Boulder", "Cascade", "Thunder", "Rainbow", "Soul", "Marsh", "Volcano", "Earth"]
    earned = [n for i, n in enumerate(names) if byte_val & (1 << i)]
    return f"{count_badges(byte_val)}/8 [{', '.join(earned) or 'none'}]"


def format_battle(val: int) -> str:
    return {0: "overworld", 1: "wild", 2: "trainer"}.get(val, f"?({val})")


def format_player_state(val: int) -> str:
    return {0: "normal", 1: "bike", 2: "surf"}.get(val, f"state_{val}")


def run(rom_path: str, state_path: str | None, speed: int, ticks: int):
    host = PyBoyHost(rom_path, headless=False, speed=speed, cgb=True)
    host.start()

    if state_path:
        host.load_state_from_file(state_path)
        print(f"Loaded save state: {state_path}")

    addr_lookup = {f.name: (f.address, f.normalize_max) for f in POKEMON_GOLD_RAM}
    step = 0
    start_time = time.monotonic()

    def read_addr(addr: int) -> int:
        if 0xD000 <= addr < 0xE000:
            return host.pyboy.memory[1, addr]
        return host.pyboy.memory[addr]

    try:
        while True:
            alive = host.tick(count=ticks, render=True)
            if not alive:
                print("Emulator closed.")
                break

            vals = {name: read_addr(addr) for name, (addr, _) in addr_lookup.items()}

            hp = decode_hp(vals["party1_hp_hi"], vals["party1_hp_lo"])
            max_hp = decode_hp(vals["party1_maxhp_hi"], vals["party1_maxhp_lo"])
            money = decode_bcd_money(vals["money_hi"], vals["money_mid"], vals["money_lo"])
            elapsed = time.monotonic() - start_time

            clear_screen()
            print("=" * 60)
            print(f"  POKEMON GOLD RAM DEBUG   step={step}  t={elapsed:.1f}s")
            print("=" * 60)
            print()

            print("  POSITION")
            print(f"    Map:       group={vals['map_group']} number={vals['map_number']}")
            print(f"    Coords:    ({vals['player_x']}, {vals['player_y']})")
            print(f"    Direction: {format_direction(vals['player_direction'])}")
            print(f"    Facing:    {format_facing(vals['facing_direction'])}")
            print(f"    Tile:      0x{vals['standing_tile']:02X}")
            print()

            print("  BADGES")
            print(f"    Johto:     {format_badges(vals['johto_badges'])}")
            print(f"    Kanto:     {format_kanto_badges(vals['kanto_badges'])}")
            print()

            print("  PARTY")
            print(f"    Size:      {vals['party_size']}")
            print(f"    Mon1 HP:   {hp}/{max_hp}" + (f" ({hp/max_hp*100:.0f}%)" if max_hp > 0 else ""))
            print(f"    Mon1 Lv:   {vals['party1_level']}")
            print()

            print("  BATTLE")
            print(f"    Mode:      {format_battle(vals['battle_mode'])}")
            print(f"    Type:      {vals['battle_type']}")
            print()

            print(f"  MONEY:       ${money:,}")
            print(f"  STEPS:       {vals['step_count']}")
            print(f"  STATE:       {format_player_state(vals['player_state'])}")
            print(f"  ACTION:      {vals['player_action']}")
            print(f"  TEXTBOX:     {vals['textbox_flags']}")
            print(f"  MENU:        {vals['menu_flags']}")
            print()

            print("-" * 60)
            print("  RAW VALUES (hex)")
            for name, (addr, _) in addr_lookup.items():
                print(f"    {name:22s}  0x{addr:04X} = 0x{vals[name]:02X} ({vals[name]:3d})")

            step += 1

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        host.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokemon Gold RAM debug viewer")
    parser.add_argument("--rom", default=ROM, help="Path to Pokemon Gold ROM")
    parser.add_argument("--state", default=None, help="Save state file to load")
    parser.add_argument("--speed", type=int, default=1, help="Emulation speed (0=uncapped)")
    parser.add_argument("--ticks", type=int, default=4, help="Frames between reads")
    args = parser.parse_args()
    run(args.rom, args.state, args.speed, args.ticks)
