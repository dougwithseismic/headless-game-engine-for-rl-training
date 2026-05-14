# Game Bridge Framework

Generic framework for training RL agents on **any** external game. Compose three abstractions — how you send actions, how you read state, how you restart episodes — and the framework gives you a standard Gymnasium environment ready for PPO/BC/curriculum training.

## Architecture

```
Training Pipeline (PPO, BC, curriculum)
         │
    gym.Env interface
         │
    ┌────┴────┐
    │         │
BaseGhostLobbyGym    ExternalGameGym
(Rust engine, PyO3)   (any external game)
    │                      │
 Scenario trait        GameBridge
                    ┌──────┼──────┐
              ActionSink  ObsSource  ResetStrategy
```

Two equal paths to training:
- **Headless Rust engine** — 19K+ FPS, for scenarios built in Bevy/Rapier
- **External game bridge** — any game with the right I/O adapters

Both produce standard `gym.Env` objects. The training pipeline doesn't care which one it gets.

## Core Abstractions

### ActionSink — how you send actions to the game

```python
class ActionSink(Protocol):
    def info(self) -> ActionSinkInfo: ...
    def connect(self) -> None: ...
    def send(self, action: np.ndarray) -> None: ...
    def reset(self) -> None: ...
    def disconnect(self) -> None: ...
```

Implementations:
- `GamepadActionSink` — wraps the existing Xbox 360 `GamepadBackend` (for Assetto Corsa, PC games via ViGEmBus)
- `PyBoyActionSink` — Game Boy button presses via PyBoy emulator API
- `MockActionSink` — records actions in a ring buffer (testing)

### ObservationSource — how you read game state

```python
class ObservationSource(Protocol):
    def info(self) -> ObservationSourceInfo: ...
    def connect(self) -> None: ...
    def read(self) -> np.ndarray: ...
    def is_terminal(self) -> bool: ...
    def disconnect(self) -> None: ...
```

Implementations:
- `PyBoyObservationSource` — RAM reads + optional screen capture from PyBoy
- `MockObservationSource` — random/scripted observations (testing)

### ResetStrategy — how you restart episodes

```python
class ResetStrategy(Protocol):
    def info(self) -> ResetInfo: ...
    def reset(self) -> None: ...
    def set_checkpoint(self, checkpoint_id: str) -> None: ...
```

Implementations:
- `PyBoyReset` — instant save state load (~5ms), supports named checkpoints for curriculum
- `MockReset` — no-op (testing)

## GameBridge

Composes all three into a coherent game interface:

```python
bridge = GameBridge(
    action_sink=my_sink,
    observation_source=my_source,
    reset_strategy=my_reset,
    config=GameBridgeConfig(
        name="my_game",
        timing=TimingConfig(policy=TimingPolicy.FREE_RUNNING),
    ),
)
```

Timing policies:
- `FREE_RUNNING` — step as fast as possible (emulators, training)
- `REAL_TIME` — sleep to maintain target Hz (live games like Assetto Corsa at 20Hz)
- `FIXED_STEP` — external clock (robotics)

## ExternalGameGym

Wraps any `GameBridge` into a standard Gymnasium environment:

```python
env = ExternalGameGym(
    bridge=bridge,
    reward_fn=my_reward_function,  # (prev_obs, action, obs) -> float
    max_steps=10_000,
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

Reward functions are pluggable Python callables — no need to modify the game.

## Named Observations

Observation sources can declare feature names, making reward functions readable and resilient to reordering:

```python
# The bridge builds a feature_index from the source's feature names
bridge = make_pokemon_red_bridge("pokemon_red.gb")
idx = bridge.feature_index  # {"player_x": 0, "player_y": 1, "map_id": 2, "badges": 4, ...}

# Reward function uses names instead of magic numbers
def reward_fn(prev_obs, action, obs):
    badges = int(obs[idx["badges"]] * 8)
    new_x = obs[idx["player_x"]]
    return float(badges) + float(abs(new_x - prev_obs[idx["player_x"]]))

env = ExternalGameGym(bridge=bridge, reward_fn=reward_fn)
env.feature_index  # also available on the gym
```

For custom observation sources, return `feature_names` and optionally `feature_groups` from `info()`:

```python
from bridges.core.obs_source import ObservationSourceInfo, FeatureGroup

class AoE2ObsSource:
    def info(self):
        return ObservationSourceInfo(
            name="aoe2",
            observation_space=Box(0, 1, shape=(12,)),
            native_hz=None,
            platform="any",
            feature_names=[
                "gold", "food", "wood", "stone",       # resources (0-3)
                "villagers", "military", "buildings",   # economy (4-6)
                "enemy_mil", "enemy_dist", "enemy_age", # enemy (7-9)
                "current_age", "pop_headroom",          # misc (10-11)
            ],
            feature_groups=[
                FeatureGroup("resources", start=0, length=4),
                FeatureGroup("economy", start=4, length=3),
                FeatureGroup("enemy", start=7, length=3),
                FeatureGroup("misc", start=10, length=2),
            ],
        )
```

Then in reward functions:

```python
bridge = make_aoe2_bridge(...)
idx = bridge.feature_index    # {"gold": 0, "food": 1, ...}
groups = bridge.feature_groups  # {"resources": slice(0, 4), "economy": slice(4, 7), ...}

def reward_fn(prev_obs, action, obs):
    resources = obs[groups["resources"]]       # array of 4 values
    total_resources = float(resources.sum())
    military = obs[idx["military"]]            # single value
    return total_resources + military * 2.0
```

## Save States

For emulator-based games, save states enable instant deterministic resets:

```python
# Boot the game once, skip intros, get to gameplay
host = PyBoyHost("pokemon_gold.gbc", headless=True, speed=0, cgb=True)
host.start()
# ... advance past menus ...

# Save to disk (reuse across sessions forever)
host.save_state_to_file("data/pokemon_gold_ready.state")

# Every episode reset loads the save state (~5ms)
# No more sitting through title screens
```

Save states also enable curriculum learning — save checkpoints at different game states and switch between them:

```python
reset = PyBoyReset(host)
reset.save_checkpoint("pallet_town")
# ... advance to next area ...
reset.save_checkpoint("viridian_city")

# Train on progressively harder checkpoints
reset.set_checkpoint("viridian_city")
```

## Concurrency

Each emulator instance runs in its own process. SB3's `SubprocVecEnv` handles parallelism:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

vec_env = SubprocVecEnv([make_env for _ in range(8)])
# 8 parallel Pokemon Gold instances = ~12K steps/sec on M4 Max
```

Tested throughput (Pokemon Gold, M4 Max):

| Setup | Steps/sec |
|---|---|
| 1 emulator | ~3,000 |
| 4 parallel | ~7,400 |
| 8 parallel | ~12,600 |

Each step advances 8 emulator frames, so 8 envs = ~100K emulated frames/sec.

## Game Profiles

Pre-built configurations for specific games. Each is a factory function:

```python
from bridges.profiles.pokemon_red import make_pokemon_red_bridge, ExplorationReward

bridge = make_pokemon_red_bridge("pokemon_red.gb", headless=True, speed=0)
env = ExternalGameGym(bridge=bridge, reward_fn=ExplorationReward(), max_steps=20_000)
```

Available profiles:
- `pokemon_red` — 16 RAM features (position, HP, badges, battle state), exploration + progress rewards
- `template` — mock components, copy and customize for any new game

## Adding a New Game

1. Choose your I/O components:
   - ActionSink: gamepad? keyboard? emulator API? network?
   - ObservationSource: RAM reads? screen capture? shared memory? network telemetry?
   - ResetStrategy: save states? keyboard macro? API call?

2. If your game runs in an emulator, create a host (like `PyBoyHost`) that all three components share.

3. Create a profile in `bridges/profiles/your_game.py`:

```python
def make_your_game_bridge(rom_path, **kwargs) -> GameBridge:
    host = YourEmulatorHost(rom_path)
    return GameBridge(
        action_sink=YourActionSink(host),
        observation_source=YourObsSource(host, features=[...]),
        reset_strategy=YourReset(host),
        config=GameBridgeConfig(name="your_game", timing=...),
    )
```

4. Write a reward function: `(prev_obs, action, obs) -> float`

5. Register in `bridges/profiles/__init__.py`

## Directory Structure

```
bridges/
  core/                  # Generic abstractions
    action_sink.py       # ActionSink protocol
    obs_source.py        # ObservationSource protocol
    reset_strategy.py    # ResetStrategy protocol
    timing.py            # TimingPolicy, StepTimer
    bridge.py            # GameBridge compositor
  sinks/                 # ActionSink implementations
    gamepad_sink.py      # Xbox 360 gamepad (wraps existing GamepadBackend)
    pyboy_sink.py        # Game Boy buttons via PyBoy
    mock_sink.py         # Testing mock
  sources/               # ObservationSource implementations
    pyboy_source.py      # RAM reads + screen capture via PyBoy
    mock_source.py       # Testing mock
  resets/                # ResetStrategy implementations
    pyboy_reset.py       # Instant save state resets via PyBoy
    mock_reset.py        # Testing mock
  emulators/             # Shared emulator hosts
    pyboy_host.py        # PyBoy lifecycle, memory, save states
  profiles/              # Pre-built game configurations
    pokemon_red.py       # Pokemon Red/Blue profile + reward functions
    template.py          # Copy-and-customize template
  gamepad/               # Existing Xbox 360 gamepad abstraction
    protocol.py, mock_backend.py, vgamepad_backend.py,
    pid.py, factory.py, visualizer.py

glgym/
  gym_external.py        # ExternalGameGym (peer to BaseGhostLobbyGym)
```

## Running Tests

```bash
cd python

# All bridge tests (no ROM needed)
pytest bridges/ -v

# Tests that need a Game Boy ROM
POKEMON_ROM=/path/to/rom.gb pytest bridges/emulators/test_pyboy.py -v -m rom

# Parallel throughput test
python test_parallel.py
```
