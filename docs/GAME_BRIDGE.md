# Game Bridge Framework

Train RL agents on any external game — emulators, racing sims, PC games, robotics — using the same training pipeline (PPO, BC, curriculum) that powers GhostLobby's headless Rust scenarios.

---

## Why This Exists

GhostLobby has two paths to training:

1. **Headless Rust engine** — Bevy ECS + Rapier physics, 19K+ FPS, scenarios coded in Rust. Fast iteration, but you need to implement the game logic yourself.

2. **External game bridge** — connect to any running game (emulator, sim, PC game) via composable I/O adapters. Slower (real-time or emulator speed), but you can train on games that already exist.

Both produce standard `gym.Env` objects. The training pipeline doesn't care which one it gets. A policy trained on the headless engine can be deployed to the real game, or vice versa.

---

## Architecture

```
Training Pipeline (PPO, BC, curriculum)
         |
    gym.Env interface
         |
    +----+----+
    |         |
BaseGhostLobbyGym    ExternalGameGym
(Rust engine, PyO3)   (any external game)
    |                      |
 Scenario trait        GameBridge
                    +------+------+
              ActionSink  ObsSource  ResetStrategy
```

Every external game interaction decomposes into three orthogonal concerns:

| Abstraction | What it does | Examples |
|---|---|---|
| **ActionSink** | Send actions to the game | Gamepad, keyboard/mouse, emulator buttons, network commands |
| **ObservationSource** | Read state from the game | RAM reads, shared memory, screen capture, network telemetry |
| **ResetStrategy** | Restart episodes | Save states, keyboard macros, API calls |

A **GameBridge** composes all three. An **ExternalGameGym** wraps any GameBridge into a Gymnasium environment.

---

## Core Protocols

All protocols use `typing.Protocol` with `@runtime_checkable`. Implementations can be any class that has the right methods — no inheritance required.

### ActionSink

```python
from bridges.core.action_sink import ActionSink, ActionSinkInfo

class ActionSink(Protocol):
    def info(self) -> ActionSinkInfo: ...
    def connect(self) -> None: ...
    def send(self, action: np.ndarray) -> None: ...
    def reset(self) -> None: ...
    def disconnect(self) -> None: ...
```

`send()` takes a raw `np.ndarray` whose shape matches `info().action_space`. The implementation translates this into game-specific inputs (axis values, button presses, key combos, etc).

**Built-in implementations:**

| Class | What it does | Platform |
|---|---|---|
| `GamepadActionSink` | Wraps existing `GamepadBackend` (Xbox 360 via ViGEmBus) | Windows |
| `PyBoyActionSink` | Game Boy button presses via PyBoy API | Any |
| `MockActionSink` | Records actions in a ring buffer | Any (testing) |

### ObservationSource

```python
from bridges.core.obs_source import ObservationSource, ObservationSourceInfo

class ObservationSource(Protocol):
    def info(self) -> ObservationSourceInfo: ...
    def connect(self) -> None: ...
    def read(self) -> np.ndarray: ...
    def is_terminal(self) -> bool: ...
    def disconnect(self) -> None: ...
```

`read()` blocks until fresh data is available and returns a pre-normalised flat array. Normalisation is the implementation's job — known physical bounds for racing telemetry, fixed ranges for RAM addresses, CNN features for screen capture.

`is_terminal()` checks if the game is in a terminal state (game over, crash, etc). Called after every `read()`.

**Built-in implementations:**

| Class | What it does | Platform |
|---|---|---|
| `PyBoyObservationSource` | RAM reads + optional screen capture via PyBoy | Any |
| `MockObservationSource` | Random or scripted observations | Any (testing) |

### ResetStrategy

```python
from bridges.core.reset_strategy import ResetStrategy, ResetInfo

class ResetStrategy(Protocol):
    def info(self) -> ResetInfo: ...
    def reset(self) -> None: ...
    def set_checkpoint(self, checkpoint_id: str) -> None: ...
```

`reset()` blocks until the game is ready for a new episode. The implementation handles the full reset sequence — save state loads, keyboard macros, API calls.

`set_checkpoint()` selects which game state to reset to. Enables curriculum learning: start from progressively harder save states.

**Built-in implementations:**

| Class | What it does | Platform |
|---|---|---|
| `PyBoyReset` | Instant save state load (~5ms) | Any |
| `MockReset` | No-op instant reset | Any (testing) |

---

## GameBridge

Composes ActionSink + ObservationSource + ResetStrategy into one coherent game interface:

```python
from bridges.core.bridge import GameBridge, GameBridgeConfig
from bridges.core.timing import TimingConfig, TimingPolicy

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

### Timing Policies

| Policy | Behaviour | Use case |
|---|---|---|
| `FREE_RUNNING` | Step as fast as possible | Emulators, training |
| `REAL_TIME` | Sleep to maintain `target_hz` | Live games (Assetto Corsa at 20Hz) |
| `FIXED_STEP` | External clock (not yet implemented) | Robotics |

### Bridge Lifecycle

```python
# Manual
bridge.connect()
obs = bridge.reset()
obs, terminal = bridge.step(action)
bridge.disconnect()

# Context manager
with bridge:
    obs = bridge.reset()
    obs, terminal = bridge.step(action)
```

### Properties

| Property | Type | Description |
|---|---|---|
| `action_space` | `gym.Space` | The action space (cached from sink) |
| `observation_space` | `gym.Space` | The observation space (cached from source) |
| `feature_index` | `dict[str, int]` | Name-to-index mapping for individual features |
| `feature_groups` | `dict[str, slice]` | Name-to-slice mapping for feature groups |
| `connected` | `bool` | Whether the bridge is connected |

---

## ExternalGameGym

Wraps any GameBridge into a standard Gymnasium environment:

```python
from glgym.gym_external import ExternalGameGym

env = ExternalGameGym(
    bridge=bridge,
    reward_fn=my_reward_fn,   # (prev_obs, action, obs) -> float
    max_steps=2048,
    phase=1,                  # optional curriculum phase
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

### Reward Functions

Reward is computed in Python via a callable. The signature is:

```python
def reward_fn(prev_obs: np.ndarray, action: np.ndarray, obs: np.ndarray) -> float:
```

The function receives the previous observation, the action taken, and the current observation. Return a float. Use `feature_index` or `feature_groups` (see below) for readable access to specific features.

If no reward function is provided, reward is always 0.0.

### Attribute Compatibility

ExternalGameGym uses the same attribute names as BaseGhostLobbyGym so callbacks and logging code work on either:

| Attribute | Type | Description |
|---|---|---|
| `current_step` | `int` | Steps taken in the current episode |
| `episode_reward` | `float` | Accumulated reward in the current episode |
| `max_steps` | `int` | Maximum steps before truncation |
| `phase` | `int \| None` | Curriculum phase (for phase-aware reward functions) |
| `feature_index` | `dict[str, int]` | Forwarded from bridge |
| `feature_groups` | `dict[str, slice]` | Forwarded from bridge |

---

## Named Observations

Observation sources can declare feature names and groups. This makes reward functions readable, self-documenting, and resilient to feature reordering.

### Feature Index (individual features)

Each feature gets a name and an index into the flat observation array:

```python
bridge = make_pokemon_red_bridge("pokemon_red.gb")
idx = bridge.feature_index
# {"player_x": 0, "player_y": 1, "map_id": 2, "badges": 4, ...}

def reward_fn(prev_obs, action, obs):
    badges = int(obs[idx["badges"]] * 8)
    return float(badges)
```

### Feature Groups (slices of related features)

Groups let you grab a sub-array of related features at once:

```python
from bridges.core.obs_source import FeatureGroup

# In your observation source:
def info(self):
    return ObservationSourceInfo(
        ...,
        feature_names=["gold", "food", "wood", "stone", "infantry", "cavalry", "archers"],
        feature_groups=[
            FeatureGroup("resources", start=0, length=4),
            FeatureGroup("military", start=4, length=3),
        ],
    )

# In your reward function:
groups = bridge.feature_groups  # {"resources": slice(0, 4), "military": slice(4, 7)}

def reward_fn(prev_obs, action, obs):
    resources = obs[groups["resources"]]     # np.ndarray of shape (4,)
    military = obs[groups["military"]]       # np.ndarray of shape (3,)
    return float(resources.sum() + military.sum() * 2.0)
```

### Both Levels Together

Use `feature_index` for individual values and `feature_groups` for batch access in the same reward function:

```python
idx = bridge.feature_index
groups = bridge.feature_groups

def reward_fn(prev_obs, action, obs):
    # Group-level: are total resources increasing?
    prev_total = prev_obs[groups["resources"]].sum()
    curr_total = obs[groups["resources"]].sum()
    resource_growth = curr_total - prev_total

    # Individual: bonus for specific milestones
    badges = int(obs[idx["badges"]] * 8)

    return resource_growth + badges * 10.0
```

### Declaring Features in a Custom Source

Return `feature_names` and `feature_groups` from your `info()` method:

```python
from bridges.core.obs_source import ObservationSourceInfo, FeatureGroup

class MyGameObsSource:
    def info(self):
        return ObservationSourceInfo(
            name="my_game",
            observation_space=gym.spaces.Box(0, 1, shape=(12,)),
            native_hz=None,
            platform="any",
            feature_names=[
                "gold", "food", "wood", "stone",
                "villagers", "military", "buildings",
                "enemy_mil", "enemy_dist", "enemy_age",
                "current_age", "pop_headroom",
            ],
            feature_groups=[
                FeatureGroup("resources", start=0, length=4),
                FeatureGroup("economy", start=4, length=3),
                FeatureGroup("enemy", start=7, length=3),
                FeatureGroup("misc", start=10, length=2),
            ],
        )
```

Both are optional. If you don't declare them, `feature_index` and `feature_groups` are empty dicts and everything still works — you just use positional indexing.

---

## Save States

For emulator-based games, save states enable instant deterministic resets. No more sitting through title screens or boot sequences.

### One-Time Setup

Boot the game, advance past intros, save the state to disk:

```python
from bridges.emulators.pyboy_host import PyBoyHost

host = PyBoyHost("pokemon_gold.gbc", headless=True, speed=0, cgb=True)
host.start()

# Advance past title screens (mash A/Start)
for i in range(1000):
    if i % 20 < 3:
        host.button("a")
    host.tick(1, render=False)

# Save to disk -- reuse across sessions forever
host.save_state_to_file("data/pokemon_gold_ready.state")
host.stop()
```

### Using Save States in Training

Load the state file at env creation. Every `reset()` restores to that state in ~5ms:

```python
host = PyBoyHost("pokemon_gold.gbc", headless=True, speed=0, cgb=True)
host.start()
host.load_state_from_file("data/pokemon_gold_ready.state")

reset = PyBoyReset(host, auto_save_initial=True)
# First reset() saves the current state as "default"
# Every subsequent reset() loads "default" instantly
```

### Multiple Checkpoints for Curriculum

Save states at different game points and switch between them:

```python
reset = PyBoyReset(host)

# Save checkpoints at different locations
reset.save_checkpoint("pallet_town")
# ... advance game ...
reset.save_checkpoint("viridian_city")
# ... advance more ...
reset.save_checkpoint("pewter_gym")

# Train on progressively harder checkpoints
reset.set_checkpoint("pallet_town")   # easy
# ... after agent masters this area ...
reset.set_checkpoint("viridian_city") # harder
```

---

## Concurrency

Each emulator runs in its own process. SB3's `SubprocVecEnv` handles parallelism automatically.

### Setup

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    host = PyBoyHost("pokemon_gold.gbc", headless=True, speed=0, cgb=True)
    host.start()
    host.load_state_from_file("data/pokemon_gold_ready.state")
    # ... build bridge and env ...
    return ExternalGameGym(bridge=bridge, reward_fn=reward, max_steps=2048)

# 8 parallel emulators
vec_env = SubprocVecEnv([make_env for _ in range(8)])
```

### Throughput (M4 Max, Pokemon Gold)

| Setup | Steps/sec | vs Single |
|---|---|---|
| 1 emulator | ~3,000 | 1x |
| 2 parallel | ~4,100 | 1.4x |
| 4 parallel | ~7,400 | 2.5x |
| 8 parallel | ~12,600 | 4.2x |

Each step advances 8 emulator frames. 8 envs = ~100K emulated frames/sec.

**Important:** The `make_env` function must be defined at module level (not inside `if __name__ == "__main__"` or a REPL) for `SubprocVecEnv` to pickle it correctly. See `python/test_parallel.py` for a working example.

---

## PyBoy Integration

The Game Boy Color emulator integration is the first complete game bridge. It demonstrates the full pattern.

### Components

| Component | File | Description |
|---|---|---|
| `PyBoyHost` | `bridges/emulators/pyboy_host.py` | Shared emulator instance — lifecycle, memory, save states, screen |
| `PyBoyActionSink` | `bridges/sinks/pyboy_sink.py` | Maps `MultiBinary(8)` to button presses |
| `PyBoyObservationSource` | `bridges/sources/pyboy_source.py` | RAM reads + optional screen capture |
| `PyBoyReset` | `bridges/resets/pyboy_reset.py` | Save state resets with named checkpoints |

### Button Mapping

The action space is `MultiBinary(8)`:

| Index | Button |
|---|---|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |
| 4 | A |
| 5 | B |
| 6 | Start |
| 7 | Select |

### RAM Observations

Pass a list of `RAMFeature(name, address, normalize_max)` to read specific memory addresses:

```python
from bridges.sources.pyboy_source import PyBoyObservationSource, RAMFeature

source = PyBoyObservationSource(
    host=host,
    ram_features=[
        RAMFeature("player_x", 0xD362, 255.0),
        RAMFeature("player_y", 0xD361, 255.0),
        RAMFeature("badges",   0xD356, 8.0),
    ],
    ticks_per_step=24,    # 24 frames per agent step
    render=False,         # no screen rendering (faster)
)
```

Values are read as unsigned bytes (0-255) and divided by `normalize_max` to produce `[0, 1]` floats.

### Screen Capture

Add `include_screen=True` to append downscaled grayscale screen pixels to the observation:

```python
source = PyBoyObservationSource(
    host=host,
    ram_features=[...],
    include_screen=True,
    screen_downscale=4,   # 160x144 -> 40x36 = 1440 pixels
    render=True,          # must be True for screen capture
)
# Observation: [ram_features..., pixel_0, pixel_1, ..., pixel_1439]
```

### Game Boy Color

Set `cgb=True` on the host for Game Boy Color ROMs:

```python
host = PyBoyHost("pokemon_gold.gbc", headless=True, speed=0, cgb=True)
```

### Headed Mode

Set `headless=False` and `speed=1` to watch the agent play in a window:

```python
host = PyBoyHost("pokemon_gold.gbc", headless=False, speed=1, cgb=True)
```

---

## Adding a New Game

### Step 1: Choose Your I/O

Decide which ActionSink, ObservationSource, and ResetStrategy to use:

| Game Type | ActionSink | ObservationSource | ResetStrategy |
|---|---|---|---|
| Game Boy / retro emulator | `PyBoyActionSink` | `PyBoyObservationSource` (RAM) | `PyBoyReset` (save states) |
| Racing sim (Assetto Corsa) | `GamepadActionSink` | Shared memory reader (custom) | Keyboard macro (custom) |
| PC game (Unity/Unreal) | Gamepad or keyboard (custom) | Network telemetry (custom) | Network command (custom) |
| Browser game | Keyboard via Playwright (custom) | DOM/screen capture (custom) | Page reload (custom) |

If a sink/source/reset doesn't exist yet, implement the protocol — it's 4-6 methods.

### Step 2: Create a Profile

A profile is a factory function that composes the three components:

```python
# bridges/profiles/my_game.py

from bridges.core.bridge import GameBridge, GameBridgeConfig
from bridges.core.timing import TimingConfig, TimingPolicy

def make_my_game_bridge(**kwargs) -> GameBridge:
    # Create your components
    sink = MyActionSink(...)
    source = MyObsSource(...)
    reset = MyReset(...)

    return GameBridge(
        action_sink=sink,
        observation_source=source,
        reset_strategy=reset,
        config=GameBridgeConfig(
            name="my_game",
            timing=TimingConfig(policy=TimingPolicy.FREE_RUNNING),
        ),
    )
```

### Step 3: Write a Reward Function

```python
def make_reward_fn(feature_index, feature_groups):
    idx = feature_index
    groups = feature_groups

    def reward_fn(prev_obs, action, obs):
        # Your reward logic here
        return float(obs[idx["score"]] - prev_obs[idx["score"]])

    return reward_fn
```

### Step 4: Wire It Up

```python
from glgym.gym_external import ExternalGameGym

bridge = make_my_game_bridge(...)
reward_fn = make_reward_fn(bridge.feature_index, bridge.feature_groups)
env = ExternalGameGym(bridge=bridge, reward_fn=reward_fn, max_steps=2048)

# Standard gym interface -- works with SB3, any RL framework
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

### Step 5: Register (Optional)

Add to the bridge registry for CLI integration:

```python
# bridges/profiles/__init__.py
BRIDGE_REGISTRY["my_game"] = "bridges.profiles.my_game.make_my_game_bridge"
```

---

## File Structure

```
python/bridges/
  core/                      # Protocols and composition
    action_sink.py           # ActionSink protocol + ActionSinkInfo
    obs_source.py            # ObservationSource protocol + FeatureGroup
    reset_strategy.py        # ResetStrategy protocol + ResetInfo
    timing.py                # TimingPolicy, TimingConfig, StepTimer
    bridge.py                # GameBridge compositor
  sinks/                     # ActionSink implementations
    gamepad_sink.py          # Xbox 360 gamepad (wraps existing GamepadBackend)
    pyboy_sink.py            # Game Boy buttons via PyBoy
    mock_sink.py             # Testing
  sources/                   # ObservationSource implementations
    pyboy_source.py          # RAM reads + screen capture via PyBoy
    mock_source.py           # Testing
  resets/                    # ResetStrategy implementations
    pyboy_reset.py           # Save state resets via PyBoy
    mock_reset.py            # Testing
  emulators/                 # Shared emulator hosts
    pyboy_host.py            # PyBoy lifecycle, memory, save states
  profiles/                  # Pre-built game configurations
    pokemon_red.py           # Pokemon Red profile + reward functions
    template.py              # Copy-and-customise template
  gamepad/                   # Existing Xbox 360 gamepad abstraction (untouched)

python/glgym/
  gym_external.py            # ExternalGameGym
  gym_base.py                # BaseGhostLobbyGym (Rust engine)
```

---

## Testing

```bash
cd python

# All bridge tests (no ROM needed) -- 86 tests
pytest bridges/ -v

# Tests that need a Game Boy ROM
POKEMON_ROM=/path/to/rom.gb pytest bridges/emulators/test_pyboy.py -v -m rom

# Parallel throughput benchmark
python test_parallel.py
```

---

## Cross-Platform Development

Everything is testable on Mac with mock backends. Real hardware/games are only needed for final integration.

| Component | Mac | Windows | Notes |
|---|---|---|---|
| Core framework + mocks | Yes | Yes | No dependencies |
| PyBoy emulator | Yes | Yes | `pip install pyboy` |
| Xbox 360 gamepad (mock) | Yes | Yes | Mock backend |
| Xbox 360 gamepad (real) | No | Yes | Needs ViGEmBus driver |
| Assetto Corsa telemetry | No | Yes | Win32 shared memory |

**Development workflow:** Build and test on Mac with mocks and emulators. Deploy to Windows only for games that require it (Assetto Corsa, games needing real gamepad input).
