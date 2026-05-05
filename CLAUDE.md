# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GhostLobby -- a headless, config-driven ECS game engine in Rust for RL agent training. Uses Bevy ECS for the simulation core, Rapier2D for physics, Axum for the WebSocket/HTTP server, and PyO3/Maturin for Python bindings. Runs at 230K+ ticks/sec headless.

## Commands

```bash
# Build everything (all workspace members)
cargo build

# Run the server (default config: arena deathmatch)
cargo run --bin ghostlobby-server
cargo run --bin ghostlobby-server -- configs/oval_race.json

# Run the benchmark (1M ticks)
cargo run --example benchmark --release -p ghostlobby-engine

# Run tests (61 tests)
cargo test -p ghostlobby-engine

# Clippy (strict)
cargo clippy -p ghostlobby-engine -p ghostlobby-server -p ghostlobby-telemetry -- -D warnings

# Build the Python wheel (requires maturin + venv)
source .venv/bin/activate
cd crates/py && maturin develop --release

# Train an RL agent
python python/train.py --config configs/1v1_deathmatch.json --timesteps 500000

# Evaluate a trained model
python python/evaluate.py --model runs/<run>/final_model.zip --episodes 5
```

## Architecture

Cargo workspace with 4 crates:

- **`crates/engine`** -- Core simulation. Bevy ECS with Rapier2D physics. Modular system schedule using `EnginePhase` sets: `ClearBuffers -> AiDecisions -> PrePhysics -> PhysicsStep -> PostPhysics -> GameLogic -> StateTransitions -> Telemetry`. Entry point is `TickRunner` (direct via `::new()` or customisable via `EngineBuilder`).
- **`crates/telemetry`** -- Sink trait (`TelemetrySink`) with implementations: `WsSink` (broadcast channel -> WebSocket clients), `FileSink` (append JSONL), `BufferSink` (in-memory ring for Python).
- **`crates/server`** -- Axum HTTP/WS server. Spawns a `TickRunner` in a tokio task, exposes `/api/*` endpoints and `/ws/observe` + `/ws/play`. Serves `web/` as static files.
- **`crates/py`** -- PyO3 module (`ghostlobby`). Exposes `GhostLobbyEnv` with `reset()`/`step()` for Gymnasium-style RL training. Supports multiple scenarios.

## Key Patterns

**Scenario system** -- Game modes implement the `Scenario` trait (8 methods: name, action_space, observation_space, setup, register_systems, observe, reward, is_done). Each scenario is a self-contained game definition. Current scenarios: `DeathmatchScenario`, `MobaLaneScenario`, `RacingScenario`.

**Action system** -- Actions are flat `Vec<f32>` arrays with multi-head definitions (`ActionSpaceDef`). Each head is either `Continuous { size, low, high }` or `Discrete { n }`. `RawActionBuffer` stores per-entity actions. `ActionMaskBuffer` provides per-tick validity masks.

**Observation system** -- Scenarios implement `observe()` which fills an `ObsWriter` with named feature arrays (self_features, entities, action_mask, etc.). `ObservationSpaceDef` describes shapes. `AgentRegistry` maps agent indices to ECS entities.

**Scripted AI** -- `ScriptedAi` component holds a closure (`AiFn`) that produces `ActionDict` from `AiContext`. Built-in AIs: `aggressive_ai()`, `creep_ai()`, `passive_ai()`, `racing_ai()`. Same action format as RL agents.

**Physics** -- `PhysicsState` wraps Rapier2D directly (not bevy_rapier). `PhysicsHandle` component links ECS entities to Rapier rigid bodies/colliders. Core systems sync between ECS Position/Velocity and Rapier state.

**Config-driven** -- All game params live in JSON (`configs/`). `GameConfig` has typed fields for arena/movement/combat/spawning/teams/obstacles, plus a `serde_json::Value` `extra` field for scenario-specific extensions.

**Telemetry** -- `TelemetryBuffer` resource collects events per tick. Events: `WorldSnapshot`, `Damage`, `Kill`, `Spawn`, `ShotFired`, `TickComplete`.

## Server Endpoints

- `GET /api/health` -- liveness
- `GET /api/match` -- current tick, title, status
- `POST /api/match/reset` -- reset simulation
- `GET /api/config` -- current GameConfig as JSON
- `GET /ws/observe` -- read-only telemetry WebSocket
- `GET /ws/play` -- bidirectional WebSocket (send `{ "source_id": 0, "actions": [...] }`)

## Training Pipeline

- `python/ghostlobby_gym.py` -- Gymnasium wrapper around `GhostLobbyEnv`
- `python/train.py` -- SB3 PPO training with CLI args, checkpoints, TensorBoard, eval callbacks
- `python/evaluate.py` -- Load and evaluate trained models
- Runs saved to `runs/{name}_{timestamp}/` with experiment.json, checkpoints, tensorboard logs

## Web Viewer

`web/index.html` -- standalone Canvas 2D top-down viewer. Connects to `/ws/observe`, renders entity positions, health bars, shot traces, kill feed. No build step required.
