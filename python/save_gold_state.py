"""Boot Pokemon Gold headed. Play past the title. CLOSE THE WINDOW to save state.

Usage:
    python save_gold_state.py
    python save_gold_state.py --output data/pokemon_gold_ready.state
"""

import argparse
import os

from bridges.emulators.pyboy_host import PyBoyHost
from bridges.profiles.pokemon_gold import POKEMON_GOLD_RAM

ROM = os.environ.get("POKEMON_ROM", "data/pokemon_gold.gbc")


def read_addr(host, addr):
    if 0xD000 <= addr < 0xE000:
        return host.pyboy.memory[1, addr]
    return host.pyboy.memory[addr]


def run(rom_path, output_path):
    host = PyBoyHost(rom_path, headless=False, speed=1, cgb=True)
    host.start()

    addr_lookup = {f.name: f.address for f in POKEMON_GOLD_RAM}

    print("Play through the title screens in the PyBoy window.")
    print(">>> CLOSE THE WINDOW to save state and exit. <<<\n")

    step = 0
    last_party = 0
    while True:
        alive = host.tick(count=4, render=True)
        if not alive:
            break

        step += 1
        if step % 60 == 0:
            party = read_addr(host, addr_lookup["party_size"])
            x = read_addr(host, addr_lookup["player_x"])
            y = read_addr(host, addr_lookup["player_y"])
            mg = read_addr(host, addr_lookup["map_group"])
            mn = read_addr(host, addr_lookup["map_number"])
            lvl = read_addr(host, addr_lookup["party1_level"])
            if party != last_party:
                print(f"  *** party changed to {party}! ***")
                last_party = party
            print(f"  step={step:5d}  party={party}  pos=({x},{y})  map=({mg},{mn})  lv={lvl}")

    # Window was closed — save state if in-game
    party = read_addr(host, addr_lookup["party_size"])
    if party > 0:
        host.save_state_to_file(output_path)
        lvl = read_addr(host, addr_lookup["party1_level"])
        x = read_addr(host, addr_lookup["player_x"])
        y = read_addr(host, addr_lookup["player_y"])
        print(f"\n=== SAVED to {output_path} ===")
        print(f"  Party: {party}, Level: {lvl}, Position: ({x},{y})")
    else:
        print("\nNo pokemon in party — state NOT saved.")

    host.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", default=ROM)
    parser.add_argument("--output", default="data/pokemon_gold_ready.state")
    args = parser.parse_args()
    run(args.rom, args.output)
