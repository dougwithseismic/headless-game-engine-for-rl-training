"""Test: VLM interprets Pokemon Gold screenshots via Ollama (Gemma 3).

Captures screenshots from the game and asks a vision model to describe
what it sees and suggest what to do next.

Usage:
    python test_vlm_pokemon.py
    python test_vlm_pokemon.py --model gemma3:4b
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from bridges.emulators.pyboy_host import PyBoyHost
from bridges.profiles.pokemon_gold import POKEMON_GOLD_RAM

ROM = os.environ.get("POKEMON_ROM", "data/pokemon_gold.gbc")
OLLAMA_URL = "http://localhost:11434/api/generate"


def read_addr(host, addr):
    if 0xD000 <= addr < 0xE000:
        return host.pyboy.memory[1, addr]
    return host.pyboy.memory[addr]


def screenshot_to_base64(host) -> str:
    screen = np.asarray(host.pyboy.screen.ndarray)[:, :, :3]
    img = Image.fromarray(screen)
    # Scale up 2x for better VLM readability
    img = img.resize((320, 288), Image.NEAREST)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ask_vlm(model: str, image_b64: str, ram_context: str) -> str:
    prompt = f"""You are analyzing a screenshot from Pokemon Gold (Game Boy Color).

Game state from RAM:
{ram_context}

Look at this screenshot and answer:
1. LOCATION: What area/building is this? (town name, route, interior)
2. VISIBLE: What can you see? (player, NPCs, buildings, paths, obstacles, grass)
3. STATE: Is there a textbox, menu, or battle on screen?
4. ACTION: What should the player do next to progress in the game? Give a specific button press or direction.

Be concise — one line per answer."""

    resp = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"]


def run(rom_path: str, state_path: str, model: str):
    host = PyBoyHost(rom_path, headless=True, speed=0, cgb=True)
    host.start()

    if state_path:
        host.load_state_from_file(state_path)
    host.tick(count=10, render=True)

    addr = {f.name: f.address for f in POKEMON_GOLD_RAM}

    scenarios = [
        ("Starting position", []),
        ("After walking right", [("right", 5)]),
        ("After walking down to edge of town", [("down", 8)]),
        ("After pressing A (interact)", [("a", 1)]),
    ]

    for label, moves in scenarios:
        # Reset to save state
        if state_path:
            host.load_state_from_file(state_path)
            host.tick(count=10, render=True)

        # Execute moves
        for btn, count in moves:
            for _ in range(count):
                host.button(btn, delay=1)
                host.tick(count=24, render=True)

        # Read RAM state
        x = read_addr(host, addr["player_x"])
        y = read_addr(host, addr["player_y"])
        mg = read_addr(host, addr["map_group"])
        mn = read_addr(host, addr["map_number"])
        party = read_addr(host, addr["party_size"])
        level = read_addr(host, addr["party1_level"])
        battle = read_addr(host, addr["battle_mode"])

        ram_context = (
            f"Position: ({x}, {y}) on map group={mg}, number={mn}\n"
            f"Party: {party} Pokemon, lead is Level {level}\n"
            f"Battle mode: {battle} (0=overworld, 1=wild, 2=trainer)"
        )

        # Get screenshot
        img_b64 = screenshot_to_base64(host)

        # Save screenshot for reference
        screen = np.asarray(host.pyboy.screen.ndarray)[:, :, :3]
        fname = f"recordings/vlm_{label.lower().replace(' ', '_')}.png"
        Image.fromarray(screen).save(fname)

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  {ram_context}")
        print(f"  Screenshot: {fname}")
        print(f"{'='*60}")

        # Ask VLM
        t0 = time.monotonic()
        try:
            response = ask_vlm(model, img_b64, ram_context)
            elapsed = time.monotonic() - t0
            print(f"  [{elapsed:.1f}s] VLM response:")
            for line in response.strip().split("\n"):
                print(f"    {line}")
        except Exception as e:
            print(f"  ERROR: {e}")

    host.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", default=ROM)
    parser.add_argument("--state", default="data/pokemon_gold_ready.state")
    parser.add_argument("--model", default="gemma3:4b")
    args = parser.parse_args()
    run(args.rom, args.state, args.model)
