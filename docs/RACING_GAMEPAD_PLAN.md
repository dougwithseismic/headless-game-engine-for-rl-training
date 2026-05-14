# Racing Gamepad & Steering Wheel — Implementation Plan

**Goal:** Build a virtual gamepad / steering wheel abstraction that works cross-platform (develop + test on Mac, deploy on Windows with Assetto Corsa), plus a headless Rust racing sim for fast RL iteration without needing a running game.

**Core constraint:** Mac has no ViGEmBus (vgamepad), no AC shared memory. Everything must be buildable, testable, and trainable on Mac using the headless sim and mock I/O layers. Windows is only needed for the final AC integration.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Training Pipeline (cross-platform)                         │
│  train.py / collect_demos.py / evaluate.py                  │
│                                                             │
│  ┌──────────────────────┐  ┌─────────────────────────────┐  │
│  │  RacingGym            │  │  AssettoCorsaGym            │  │
│  │  (headless Rust sim)  │  │  (external sim bridge)      │  │
│  │  Mac + Windows        │  │  Windows only               │  │
│  │  19K+ FPS             │  │  Real-time (20Hz)           │  │
│  └──────┬───────────────┘  └──────┬──────────────────────┘  │
│         │                         │                          │
│         │ PyO3 (ghostlobby)       │ gamepad + telemetry      │
│         │                         │                          │
│  ┌──────▼───────────────┐  ┌──────▼──────────────────────┐  │
│  │  RacingScenario       │  │  GamepadBackend (trait)      │  │
│  │  (Rapier3D bicycle    │  │  ├─ VGamepadBackend (Win)    │  │
│  │   model, track spline,│  │  ├─ MockGamepad (Mac/test)   │  │
│  │   tire slip, etc.)    │  │  └─ LoggingGamepad (debug)   │  │
│  │  Rust, crates/engine  │  │                              │  │
│  └──────────────────────┘  │  TelemetryBackend (trait)     │  │
│                             │  ├─ ACSharedMemory (Win)      │  │
│                             │  ├─ ReplayTelemetry (Mac)     │  │
│                             │  └─ MockTelemetry (test)      │  │
│                             │  Python, python/bridges/      │  │
│                             └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

Two parallel paths that share the same action/obs format:
- **Path A (headless):** Rust `RacingScenario` — runs on Mac, 19K+ FPS, for reward tuning, curriculum, architecture search
- **Path B (AC bridge):** Python bridge with swappable backends — develop the abstraction on Mac with mocks/replay, deploy on Windows with real vgamepad + AC shared memory

---

## Phase 1: Gamepad Abstraction Layer (Python)

**Where:** `python/bridges/gamepad/`

The virtual gamepad is a thin layer that maps `[-1, 1]` continuous actions to controller inputs. Abstract it behind a protocol so Mac dev uses a mock and Windows uses the real ViGEmBus driver.

### Files

```
python/bridges/__init__.py
python/bridges/gamepad/__init__.py
python/bridges/gamepad/protocol.py       # GamepadBackend Protocol class
python/bridges/gamepad/vgamepad_backend.py  # Real vgamepad (Windows)
python/bridges/gamepad/mock_backend.py    # Mock for Mac / tests
python/bridges/gamepad/factory.py        # auto-detect platform, return correct backend
```

### `protocol.py` — The Interface

```python
from typing import Protocol

class GamepadBackend(Protocol):
    def connect(self) -> None: ...
    def send(self, steering: float, acceleration: float) -> None:
        """
        steering:     [-1, 1]  full left to full right
        acceleration: [-1, 1]  -1 = full brake, 0 = coast, +1 = full throttle
        """
        ...
    def disconnect(self) -> None: ...
```

The 2D combined action space from the spec (Section 5.1). `send()` splits acceleration into throttle/brake internally.

### `vgamepad_backend.py` — Windows Real Controller

Maps to Xbox 360 via ViGEmBus:
- `steering` → left joystick X axis
- `acceleration >= 0` → right trigger (throttle)
- `acceleration < 0` → left trigger (brake)

Depends on `vgamepad` (pip install, Windows-only).

### `mock_backend.py` — Mac Development

Logs inputs to a ring buffer, provides `get_history()` for assertions in tests and a live matplotlib/terminal visualization mode for debugging. No system dependencies.

### `factory.py` — Platform Auto-Detection

```python
import platform

def make_gamepad(backend: str = "auto") -> GamepadBackend:
    if backend == "auto":
        backend = "vgamepad" if platform.system() == "Windows" else "mock"
    ...
```

### Tests (Mac-runnable)

- `python/bridges/gamepad/test_gamepad.py`
- Verify mock records history, factory returns mock on Mac
- Verify action clamping, throttle/brake split math
- Verify rate-limiting (max_delta = 0.1 per step at 20Hz)

### Deliverable

From Mac terminal:
```bash
python -c "
from bridges.gamepad.factory import make_gamepad
gp = make_gamepad()  # auto-detects mock on Mac
gp.connect()
gp.send(steering=0.3, acceleration=0.8)
gp.send(steering=0.35, acceleration=-0.5)
print(gp.get_history())  # [{steering: 0.3, throttle: 0.8, brake: 0.0}, ...]
gp.disconnect()
"
```

---

## Phase 2: Telemetry Reader Abstraction (Python)

**Where:** `python/bridges/telemetry/`

Same pattern as gamepad — abstract behind a protocol, swap real AC shared memory for a replay-from-file backend on Mac.

### Files

```
python/bridges/telemetry/__init__.py
python/bridges/telemetry/protocol.py        # TelemetryBackend Protocol
python/bridges/telemetry/schema.py          # CarState, TrackState dataclasses (~44 features)
python/bridges/telemetry/ac_backend.py      # AC shared memory reader (Windows, ctypes + mmap)
python/bridges/telemetry/replay_backend.py  # Replay from .npz recording (Mac)
python/bridges/telemetry/mock_backend.py    # Synthetic data for tests
python/bridges/telemetry/factory.py         # Platform auto-detect
python/bridges/telemetry/normalizer.py      # Raw telemetry → [-1, 1] observation vector
```

### `schema.py` — Observation Dataclass

Matches the 44-feature spec (Section 4.2):

```python
@dataclass
class CarState:
    speed_ms: float
    local_vel_x: float       # lateral
    local_vel_z: float       # longitudinal
    accel_x: float           # lateral G
    accel_z: float           # longitudinal G
    yaw_rate: float
    steer_angle: float
    throttle: float
    brake: float
    gear: int
    rpm: float
    slip_front: float
    slip_rear: float
    tyre_grip: float

@dataclass
class TrackState:
    progress: float           # 0-1 normalized track position
    cross_track_error: float  # lateral offset from centerline
    heading_error: float      # angle to track direction
    dist_left: float
    dist_right: float
    curvature: float

@dataclass
class LookaheadPoint:
    curvature: float
    width: float
    heading_delta: float
    distance: float

@dataclass
class TelemetryFrame:
    car: CarState
    track: TrackState
    lookahead: list[LookaheadPoint]  # N=5
    prev_actions: list[float]         # last 2 steps (4 values)
    packet_id: int
    timestamp: float
```

### `replay_backend.py` — Mac Development Workhorse

Loads a `.npz` file of recorded telemetry (captured once on Windows) and replays it frame-by-frame. Supports:
- Sequential replay (mimics real-time)
- Random-access by frame index
- Loop mode (wraps around at end of recording)

This means you can record 10 laps on Windows once, copy the `.npz` to Mac, and develop the entire observation pipeline, normalization, and reward function without touching Windows again.

### `normalizer.py`

Converts raw `TelemetryFrame` → flat `np.ndarray` of shape `(44,)` with `[-1, 1]` range. Uses the physical bounds from the spec (max_speed, max_g, max_slip, etc.) — no learned running stats.

### Tests (Mac-runnable)

- Schema round-trip (dataclass → numpy → dataclass)
- Normalizer bounds verification (all outputs in [-1, 1])
- Replay backend loops correctly, handles stale packet_id
- Mock backend generates plausible synthetic data

### Deliverable

```bash
# Record on Windows (one-time):
python scripts/record_ac_telemetry.py --laps 10 --output data/monza_10laps.npz

# Develop on Mac:
python -c "
from bridges.telemetry.factory import make_telemetry
from bridges.telemetry.normalizer import normalize
t = make_telemetry(replay_file='data/monza_10laps.npz')
t.connect()
frame = t.read()
obs = normalize(frame)
print(obs.shape, obs.min(), obs.max())  # (44,) ~-1.0 ~1.0
"
```

---

## Phase 3: Track Data Parser

**Where:** `python/bridges/track/`

Parse Assetto Corsa's `fast_lane.ai` files into centerline + curvature + width data. This is a one-time offline extraction per track, fully cross-platform (just file parsing).

### Files

```
python/bridges/track/__init__.py
python/bridges/track/parser.py        # Parse fast_lane.ai binary format → TrackSpline
python/bridges/track/spline.py        # TrackSpline class: query curvature, width, nearest point at any s
python/bridges/track/lookahead.py     # Generate N lookahead points from current position + speed
```

### `spline.py` — Core Data Structure

```python
@dataclass
class TrackSpline:
    centerline: np.ndarray   # (N, 3) xyz points
    left_edge: np.ndarray    # (N, 3)
    right_edge: np.ndarray   # (N, 3)
    curvature: np.ndarray    # (N,) precomputed
    width: np.ndarray        # (N,)
    total_length: float

    def nearest(self, pos: np.ndarray) -> tuple[float, float, float]:
        """Returns (s, cross_track_error, heading_error) for a world position."""
        ...

    def lookahead(self, s: float, speed: float, n_points: int = 5) -> list[LookaheadPoint]:
        """Sample N points ahead, spaced by time-to-reach (not fixed distance)."""
        ...
```

### Data Files

```
data/tracks/
  monza.npz           # Pre-extracted from fast_lane.ai
  spa.npz
```

Ship the extracted `.npz` files in the repo so Mac development never needs the raw AC files.

### Tests

- Parser round-trips a known track correctly
- Spline nearest-point query returns sane values
- Lookahead spacing scales with speed
- Curvature matches known corners (e.g., Monza T1 should be high curvature)

---

## Phase 4: Gymnasium Wrapper

**Where:** `python/glgym/gym_racing.py`

Wire the gamepad + telemetry + track into a standard Gymnasium env so the existing `train.py` / `evaluate.py` scripts work unchanged.

### Two Modes

**Mode A: Headless (Mac + Windows)**
Wraps `GhostLobbyEnv(scenario="racing")` via PyO3. Same pattern as `CsLiteGym`. Runs at 19K+ FPS, no external dependencies.

**Mode B: AC Bridge (Windows, or Mac with replay)**
Wraps the gamepad + telemetry backends directly. Runs at 20Hz real-time when connected to AC, or at replay speed with mock gamepad on Mac.

```python
class RacingGym(BaseGhostLobbyGym):
    """Headless racing via Rust engine. Fast iteration on Mac/Windows."""

    def __init__(self, config_path, ...):
        super().__init__(config_path=config_path, scenario="racing", ...)

    def _remap_actions(self, action):
        # 2D combined → steer, throttle, brake
        ...

class AssettoCorsaGym(gym.Env):
    """Bridge to running Assetto Corsa instance. Real-time 20Hz."""

    observation_space = Box(low=-1, high=1, shape=(44,))
    action_space = Box(low=-1, high=1, shape=(2,))

    def __init__(self, track, car, config, backend="auto"):
        self.gamepad = make_gamepad(backend)
        self.telemetry = make_telemetry(backend, ...)
        self.track = TrackSpline.load(f"data/tracks/{track}.npz")
        self.step_freq = 20  # Hz
        ...

    def step(self, action):
        self.gamepad.send(steering=action[0], acceleration=action[1])
        self._wait_for_next_tick()
        frame = self.telemetry.read()
        obs = normalize(frame)
        reward = self._compute_reward(frame)
        terminated = self._check_terminal(frame)
        return obs, reward, terminated, False, {}

    def reset(self, **kwargs):
        self._restart_session()
        return self._read_obs(), {}
```

### Tests (Mac-runnable)

- `RacingGym` step/reset cycle with headless engine
- `AssettoCorsaGym` with mock backends — verify obs shape, reward sign, episode termination
- Action rate-limiting enforcement
- Reward components: progress reward dominates, penalties are small

---

## Phase 5: Headless Racing Scenario (Rust)

**Where:** `crates/engine/src/scenarios/racing.rs`

A simplified vehicle sim in Rapier3D so you can train millions of steps on Mac without needing AC. Not physics-perfect — just good enough to learn steering/throttle coordination, track following, and reward shaping before transferring to AC.

### New Files

```
crates/engine/src/scenarios/racing.rs          # RacingScenario (Scenario trait impl)
crates/engine/src/scenarios/racing_track.rs    # Track spline resource, spawn points, checkpoints
crates/engine/src/scenarios/racing_vehicle.rs  # Bicycle model vehicle dynamics
configs/racing/oval_simple.json                # Dead-simple oval for initial testing
configs/racing/monza_approx.json               # Approximate Monza layout
```

### Vehicle Model

Bicycle model (not full tire sim — that's AC's job):
- Front/rear axle, wheelbase, mass, max_steer_angle
- Pacejka-lite tire model (slip angle → lateral force, one tunable parameter)
- Longitudinal: simple force = throttle * max_force - brake * max_brake - drag * v^2
- No suspension, no camber, no aero downforce — keep it trainable

### Components

```rust
#[derive(Component)]
pub struct Vehicle {
    pub wheelbase: f32,
    pub mass: f32,
    pub max_steer: f32,
    pub max_throttle_force: f32,
    pub max_brake_force: f32,
    pub drag_coeff: f32,
}

#[derive(Component)]
pub struct TrackProgress {
    pub s: f32,              // distance along spline
    pub cross_track: f32,    // lateral offset
    pub heading_error: f32,
    pub lap: u32,
    pub best_lap_time: f32,
    pub current_lap_time: f32,
}

#[derive(Resource)]
pub struct TrackDefinition {
    pub centerline: Vec<Vec3>,
    pub widths: Vec<f32>,
    pub curvatures: Vec<f32>,
    pub total_length: f32,
}
```

### Action Space

```rust
fn action_space(&self, _config: &GameConfig) -> ActionSpaceDef {
    ActionSpaceDef::new(vec![
        ActionHead::Continuous {
            name: "steering".into(),
            size: 1,
            low: vec![-1.0],
            high: vec![1.0],
        },
        ActionHead::Continuous {
            name: "acceleration".into(),
            size: 1,
            low: vec![-1.0],
            high: vec![1.0],
        },
    ])
}
```

### Observation Space

Mirror the 44-feature Python spec so policies transfer between headless and AC:
- 14 car state features
- 6 track-relative features
- 20 lookahead features (5 points x 4 features)
- 4 previous action features

### Reward Function

From spec Section 6:
```rust
let r_progress = v_along_track / v_max;
let r_offtrack = if tyres_out >= 4 { -1.0 } else { 0.0 };
let r_jerk = -alpha * (steer - prev_steer).abs();
let r_slip = -beta * (slip - slip_threshold).max(0.0);
let reward = r_progress + r_offtrack + r_jerk + r_slip;
```

### Registration

Add to `crates/engine/src/scenarios/mod.rs`:
```rust
pub mod racing;
```

Add to `crates/py/src/lib.rs` `make_scenario()`:
```rust
"racing" | "race" | "time-trial" => Ok(Box::new(RacingScenario::default())),
```

### Tests

- `crates/engine/tests/racing.rs` — vehicle drives forward, track progress increases, lap detection works
- Verify action space is `(2,)` continuous
- Verify obs space matches expected 44 features

---

## Phase 6: Configs

**Where:** `configs/racing/`

```
configs/racing/
  oval_simple.json         # Flat oval, wide, slow car — Phase 1 curriculum
  monza_approx.json        # Approximate Monza corners and widths
  curriculum.yaml          # Wide/slow → narrow/fast progression
  reward_weights.json      # Tunable reward component weights
```

### `oval_simple.json` Example

```json
{
  "title": "racing_oval_simple",
  "tick_rate": 60,
  "arena": { "width": 1000, "height": 600 },
  "teams": { "count": 1, "players_per_team": 1 },
  "extra": {
    "track": "oval_simple",
    "car": {
      "wheelbase": 2.5,
      "mass": 1200,
      "max_steer_deg": 30,
      "max_throttle_force": 8000,
      "max_brake_force": 12000,
      "drag_coeff": 0.35
    },
    "reward": {
      "progress_weight": 1.0,
      "offtrack_penalty": -1.0,
      "jerk_alpha": 0.02,
      "slip_beta": 0.1,
      "slip_threshold": 0.1
    },
    "action_rate_limit": 0.1,
    "lookahead_points": 5,
    "max_speed_ms": 80
  }
}
```

---

## Phase 7: Web Viewer (Optional — nice for demos)

**Where:** `web-app/src/scenarios/racing/`

```
web-app/src/scenarios/racing/
  index.ts              # ScenarioDefinition export
  RacingCanvas.tsx      # Top-down track view with car position + racing line
  RacingSidebar.tsx     # Live telemetry: steering/throttle/brake traces, slip, lap times
  store.ts             # Zustand store for racing telemetry events
```

Register in `web-app/src/scenarios/registry.ts`.

New telemetry event type in `crates/engine/src/telemetry.rs`:
```rust
RacingState {
    tick: u64,
    car_pos: [f32; 3],
    car_vel: [f32; 3],
    steering: f32,
    throttle: f32,
    brake: f32,
    slip_front: f32,
    slip_rear: f32,
    track_progress: f32,
    lap: u32,
    lap_time: f32,
}
```

---

## File System Summary

```
crates/engine/src/scenarios/
  mod.rs                          # + pub mod racing;
  racing.rs                       # RacingScenario (Scenario trait)
  racing_track.rs                 # TrackDefinition, spline queries
  racing_vehicle.rs               # Bicycle model, tire forces
  cs_lite.rs                      # (existing)
  tactical_deathmatch.rs          # (existing)

crates/engine/tests/
  racing.rs                       # Integration tests

crates/py/src/
  lib.rs                          # + "racing" scenario match arm

python/bridges/
  __init__.py
  gamepad/
    __init__.py
    protocol.py                   # GamepadBackend Protocol
    vgamepad_backend.py           # Windows real controller
    mock_backend.py               # Mac mock + history
    factory.py                    # Platform auto-detect
    test_gamepad.py
  telemetry/
    __init__.py
    protocol.py                   # TelemetryBackend Protocol
    schema.py                     # CarState, TrackState, TelemetryFrame
    ac_backend.py                 # AC shared memory (Windows)
    replay_backend.py             # Replay from .npz (Mac)
    mock_backend.py               # Synthetic test data
    factory.py
    normalizer.py                 # Raw → [-1,1] obs vector
    test_telemetry.py
  track/
    __init__.py
    parser.py                     # fast_lane.ai → TrackSpline
    spline.py                     # TrackSpline class
    lookahead.py                  # Lookahead point generation
    test_track.py

python/glgym/
  gym_racing.py                   # RacingGym (headless) + AssettoCorsaGym (bridge)

python/scripts/
  record_ac_telemetry.py          # One-time recording on Windows
  watch_racing.py                 # Visualize trained racing agent

configs/racing/
  oval_simple.json
  monza_approx.json
  curriculum.yaml
  reward_weights.json

data/tracks/
  oval_simple.npz                 # Pre-extracted track data
  monza.npz

web-app/src/scenarios/racing/     # (Phase 7, optional)
  index.ts
  RacingCanvas.tsx
  RacingSidebar.tsx
  store.ts
```

---

## Mac vs Windows: What Runs Where

| Component | Mac | Windows | Notes |
|-----------|-----|---------|-------|
| Gamepad mock + tests | yes | yes | `mock_backend.py` |
| Gamepad real (vgamepad) | no | yes | Needs ViGEmBus driver |
| Telemetry mock + tests | yes | yes | `mock_backend.py` |
| Telemetry replay | yes | yes | `.npz` file, no OS deps |
| Telemetry real (AC shared memory) | no | yes | Win32 mmap |
| Track parser | yes | yes | Pure file I/O |
| Headless Rust sim | yes | yes | Rapier3D, cross-platform |
| RacingGym (headless) | yes | yes | Via PyO3 |
| AssettoCorsaGym (AC bridge) | partial* | yes | *Mac with replay+mock |
| Training pipeline (BC/PPO) | yes | yes | PyTorch + SB3 |
| Web viewer | yes | yes | React app |

**Mac development loop:** Headless Rust sim → RacingGym → train with PPO → tune reward/obs → iterate at 19K FPS. Transfer policy to AC on Windows only for final validation.

---

## Build Order

### Sprint 1: Gamepad + Telemetry Abstractions (Mac-first)
1. `python/bridges/gamepad/` — protocol, mock, factory, tests
2. `python/bridges/telemetry/` — protocol, schema, mock, normalizer, tests
3. Verify everything runs on Mac with `pytest`

### Sprint 2: Headless Racing Sim (Rust)
4. `crates/engine/src/scenarios/racing_track.rs` — track spline resource
5. `crates/engine/src/scenarios/racing_vehicle.rs` — bicycle model
6. `crates/engine/src/scenarios/racing.rs` — scenario trait impl
7. `configs/racing/oval_simple.json`
8. `crates/engine/tests/racing.rs`
9. Register in `mod.rs` + `lib.rs` (PyO3)

### Sprint 3: Gymnasium + Training
10. `python/glgym/gym_racing.py` — RacingGym wrapping headless engine
11. Train PPO on oval, verify agent learns to follow track
12. Add curriculum configs, test phase advancement

### Sprint 4: Track Data + AC Bridge (needs one Windows session)
13. `python/bridges/track/` — parser, spline, lookahead
14. Record telemetry on Windows → `data/tracks/monza.npz`
15. `python/bridges/telemetry/replay_backend.py`
16. `python/bridges/telemetry/ac_backend.py`
17. `python/bridges/gamepad/vgamepad_backend.py`
18. `python/glgym/gym_racing.py` — AssettoCorsaGym mode

### Sprint 5: Integration + Polish
19. Transfer headless-trained policy to AC, validate
20. BC data collection from AC AI laps
21. Web viewer (optional)
22. Full pipeline: BC → PPO → curriculum → superhuman attempt

---

## Dependencies

### Python (cross-platform)
```
numpy
gymnasium
stable-baselines3
sb3-contrib          # TQC
torch
matplotlib           # debug viz for mock gamepad
```

### Python (Windows-only)
```
vgamepad             # ViGEmBus virtual Xbox 360
```

### Rust (already in workspace)
```
rapier3d             # already used by cs_lite
bevy_ecs
glam
serde / serde_json
```

No new Rust crate dependencies needed — the racing scenario reuses the same Rapier3D + Bevy ECS stack as cs_lite.
