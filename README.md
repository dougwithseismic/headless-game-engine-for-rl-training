# GhostLobby

Headless game engine for RL agent training. Define a game scenario, get a Gymnasium-compatible environment running at 230K+ ticks/sec.

Built in Rust (Bevy ECS + Rapier2D physics), with Python bindings via PyO3. Ships with FPS deathmatch, MOBA lane, and racing scenarios. Add your own by implementing one trait.

## Quick Start

```bash
# Build
cargo build

# Run tests (61 tests)
cargo test -p ghostlobby-engine

# Benchmark (1M ticks)
cargo run --example benchmark --release -p ghostlobby-engine

# Start server with web viewer
cargo run --bin ghostlobby-server
# Open http://localhost:3000
```

### Python / RL Training

```bash
# Set up Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy gymnasium stable-baselines3

# Build the Python module
cd crates/py && maturin develop --release && cd ../..

# Train an agent
python python/train.py --config configs/1v1_deathmatch.json --timesteps 500000

# Evaluate a trained model
python python/evaluate.py --model runs/<your_run>/final_model.zip --episodes 5

# View training dashboard
tensorboard --logdir runs/
```

### Python API

```python
import ghostlobby

env = ghostlobby.GhostLobbyEnv("configs/oval_race.json", scenario="racing")

obs, info = env.reset()
# obs = {0: {"self_features": [...], "track_waypoints": [...]}, 1: {...}, ...}

actions = {i: [0.0, 1.0, 0.0] for i in env.agents()}
obs, rewards, terminated, truncated, infos = env.step(actions)

env.action_space()        # head definitions + total_size
env.observation_space()   # feature names + shapes
env.agents()              # [0, 1, 2, 3]
```

## Architecture

```
crates/
  engine/      Core simulation -- Bevy ECS, Rapier2D physics, scenarios, observations, rewards
  server/      Axum HTTP/WebSocket server with live web viewer
  telemetry/   Event sinks (WebSocket broadcast, JSONL file, in-memory buffer)
  py/          PyO3 Python bindings -- GhostLobbyEnv

python/        Gymnasium wrapper, SB3 training script, evaluation script
configs/       Game configuration JSONs
web/           Canvas 2D top-down viewer (connects via WebSocket)
```

### Engine Phases

Each tick runs through ordered system sets. Core systems are always present; scenarios register their own into the appropriate phase.

```
ClearBuffers -> AiDecisions -> PrePhysics -> PhysicsStep -> PostPhysics -> GameLogic -> StateTransitions -> Telemetry
```

**Core systems** (engine provides): clear buffers, run scripted AI, sync actions to Rapier, step physics, sync positions back to ECS, emit telemetry snapshots.

**Scenario systems** (scenario registers): combat, death/respawn, vehicle physics, checkpoint tracking, creep spawning -- whatever the game needs.

### Scenario Trait

A scenario defines a complete game -- what entities exist, what actions are available, what the agent observes, and how rewards work.

```rust
pub trait Scenario: Send + Sync {
    fn name(&self) -> &str;
    fn action_space(&self, config: &GameConfig) -> ActionSpaceDef;
    fn observation_space(&self, config: &GameConfig) -> ObservationSpaceDef;
    fn setup(&self, world: &mut World, config: &GameConfig, physics: &mut PhysicsState);
    fn register_systems(&self, schedule: &mut Schedule);
    fn observe(&self, world: &World, agent: Entity, writer: &mut ObsWriter);
    fn reward(&self, world: &World, agent: Entity) -> f32;
    fn is_done(&self, world: &World, agent: Entity) -> bool;
}
```

### Action Spaces

Actions are flat `Vec<f32>` arrays with multi-head definitions. Each scenario defines its own layout:

| Scenario | Heads | Total |
|----------|-------|-------|
| FPS Deathmatch | `move_dir(2)`, `look_angle(1)`, `shoot(discrete 2)` | 4 |
| Racing | `steer(1)`, `throttle(1)`, `brake(discrete 2)` | 3 |
| MOBA Lane | `move_dir(2)`, `look_angle(1)`, `shoot(discrete 2)` | 4 |

RL agents send flat arrays. Scripted bots produce the same format. The engine doesn't care who's driving.

## Scenarios

### FPS Deathmatch

Team-based arena combat with hitscan weapons. Agents move, aim, and shoot. Rapier handles collision with walls and obstacles. Raycast combat with line-of-sight occlusion.

```bash
cargo run --bin ghostlobby-server -- configs/arena_deathmatch.json    # 5v5
cargo run --bin ghostlobby-server -- configs/1v1_deathmatch.json      # 1v1
```

### Racing

Oval track with vehicle physics (steering, throttle, braking). 4 checkpoints around the track, 3 laps to win. Cars collide with walls and each other via Rapier.

```bash
cargo run --bin ghostlobby-server -- configs/oval_race.json
```

### MOBA Lane

1v1 in a narrow lane with AI-controlled creep waves. Creeps spawn periodically and march toward the enemy base.

```bash
cargo run --bin ghostlobby-server -- configs/lane_lasthit.json
```

## Training

The training pipeline uses Stable Baselines3 with PPO. Each run creates a self-contained experiment directory.

```bash
python python/train.py \
  --config configs/1v1_deathmatch.json \
  --scenario fps \
  --timesteps 1000000 \
  --lr 3e-4 \
  --frame-skip 4 \
  --name my_experiment
```

This creates:

```
runs/my_experiment_2026-05-05_14-30/
  experiment.json          Full config snapshot for reproducibility
  tensorboard/             TensorBoard event logs
  checkpoints/
    model_50000_steps.zip  Periodic checkpoints
    model_100000_steps.zip
  best_model/              Best model from evaluation callbacks
  eval_logs/               Evaluation results
  final_model.zip          Final trained model
```

Resume from a checkpoint:

```bash
python python/train.py \
  --resume runs/my_experiment_2026-05-05_14-30/checkpoints/model_50000_steps.zip \
  --timesteps 500000
```

Compare experiments in TensorBoard:

```bash
tensorboard --logdir runs/
# Open http://localhost:6006
```

Evaluate a trained model:

```bash
python python/evaluate.py --model runs/my_experiment_2026-05-05_14-30/final_model.zip --episodes 10
```

## Server API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Liveness check |
| `/api/match` | GET | Current tick, title, status |
| `/api/match/reset` | POST | Reset simulation |
| `/api/config` | GET | Full game config as JSON |
| `/ws/observe` | WS | Read-only telemetry stream |
| `/ws/play` | WS | Send actions: `{"source_id": 0, "actions": [0.5, 0.3, 1.2, 0.0]}` |

## Adding a New Scenario

1. Create `crates/engine/src/scenarios/your_game.rs`
2. Define game-specific components and resources
3. Implement the `Scenario` trait (8 methods)
4. Register your systems into `EnginePhase` sets
5. Add `pub mod your_game` to `crates/engine/src/scenarios/mod.rs`
6. Add a config JSON in `configs/`
7. Add scenario detection in `crates/server/src/tick_loop.rs` and `crates/py/src/lib.rs`

No changes to the core engine required. The scenario owns its action space, observation space, game logic, rewards, and win conditions.

## Performance

Benchmark on Apple M-series (1M ticks, 10 agents, arena with obstacles):

```
Ticks/sec:   230,000+
us/tick:     ~4.3
```

Training throughput with SB3 PPO (includes Python + PyTorch overhead):

```
Steps/sec:   ~4,000 (single env)
```

## Dependencies

**Rust**: bevy_ecs 0.16, rapier2d 0.22, glam 0.29, axum 0.8, tokio, serde, pyo3 0.25

**Python**: gymnasium, stable-baselines3, numpy, maturin (build only)
