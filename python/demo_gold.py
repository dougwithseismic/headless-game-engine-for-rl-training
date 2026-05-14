"""Live demo: Pokemon Gold running through the GhostLobby bridge framework."""

import numpy as np

from bridges.emulators.pyboy_host import PyBoyHost
from bridges.sinks.pyboy_sink import PyBoyActionSink
from bridges.sources.pyboy_source import PyBoyObservationSource, RAMFeature
from bridges.resets.pyboy_reset import PyBoyReset
from bridges.core.bridge import GameBridge, GameBridgeConfig
from bridges.core.timing import TimingConfig, TimingPolicy
from glgym.gym_external import ExternalGameGym

ROM = "data/pokemon_gold.gbc"

host = PyBoyHost(ROM, headless=False, speed=1, cgb=True)

bridge = GameBridge(
    action_sink=PyBoyActionSink(host),
    observation_source=PyBoyObservationSource(
        host,
        ram_features=[
            RAMFeature("mem_c000", 0xC000, 255.0),
            RAMFeature("mem_c001", 0xC001, 255.0),
        ],
        include_screen=True,
        screen_downscale=4,
        ticks_per_step=12,
        render=True,
    ),
    reset_strategy=PyBoyReset(host),
    config=GameBridgeConfig(
        name="pokemon_gold",
        timing=TimingConfig(policy=TimingPolicy.REAL_TIME, target_hz=10),
    ),
)

env = ExternalGameGym(bridge=bridge, max_steps=600)
obs, _ = env.reset()

# Button arrays: [up, down, left, right, a, b, start, select]
IDLE = np.array([0, 0, 0, 0, 0, 0, 0, 0])
A = np.array([0, 0, 0, 0, 1, 0, 0, 0])
B = np.array([0, 0, 0, 0, 0, 1, 0, 0])
START = np.array([0, 0, 0, 0, 0, 0, 1, 0])
RIGHT = np.array([0, 0, 0, 1, 0, 0, 0, 0])
DOWN = np.array([0, 1, 0, 0, 0, 0, 0, 0])
LEFT = np.array([0, 0, 1, 0, 0, 0, 0, 0])
UP = np.array([1, 0, 0, 0, 0, 0, 0, 0])

# Boot: mash A/Start through title and menus, then wander
seq = []
seq += [IDLE] * 30
seq += [A] * 5 + [IDLE] * 15
seq += [START] * 5 + [IDLE] * 15
seq += [A] * 5 + [IDLE] * 15
seq += [A] * 5 + [IDLE] * 15
seq += [A] * 5 + [IDLE] * 15
seq += [A] * 5 + [IDLE] * 15
seq += [A] * 5 + [IDLE] * 15
seq += [A] * 5 + [IDLE] * 15
seq += [A] * 5 + [IDLE] * 15
seq += [DOWN] * 5 + [A] * 5 + [IDLE] * 15
seq += [A] * 5 + [IDLE] * 15
seq += [A] * 5 + [IDLE] * 15
# In-game movement
seq += [DOWN] * 30 + [RIGHT] * 30 + [UP] * 30 + [LEFT] * 30
seq += [DOWN] * 20 + [A] * 5 + [IDLE] * 5
seq += [RIGHT] * 30 + [UP] * 20
seq += [LEFT] * 30 + [DOWN] * 20

total_steps = min(len(seq), 500)
print(f"Pokemon Gold running — watch the window! ({total_steps} steps)")

for step in range(total_steps):
    obs, r, t, tr, info = env.step(seq[step])
    if step % 50 == 0:
        print(f"  Step {step}/{total_steps}")
    if t or tr:
        obs, _ = env.reset()

print("Done!")
env.close()
host.stop()
