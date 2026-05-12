# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GhostLobby -- a headless, config-driven ECS game engine in Rust for RL agent training. Uses Bevy ECS for the simulation core, Rapier2D/3D for physics, Axum for the WebSocket/HTTP server, and PyO3/Maturin for Python bindings. Runs at 19K+ FPS with 32 parallel envs.

## Commands

```bash
# Build everything (all workspace members)
cargo build

# Run the server (default config, port 3000)
cargo run --bin ghostlobby-server
cargo run --bin ghostlobby-server -- configs/cs_lite/cs_lite.json
cargo run --bin ghostlobby-server -- configs/tactical/tactical_open.json --port 0  # OS-assigned port

# Run the benchmark (1M ticks)
cargo run --example benchmark --release -p ghostlobby-engine

# Run tests
cargo test -p ghostlobby-engine

# Clippy (strict)
cargo clippy -p ghostlobby-engine -p ghostlobby-server -p ghostlobby-telemetry -- -D warnings

# Build the Python wheel (requires maturin + venv)
source .venv/bin/activate
cd crates/py && maturin develop --release

# Training pipeline
cd python

# Collect BC demonstrations from scripted AI
python scripts/collect_demos.py --scenario cs_lite --config configs/cs_lite/cs_lite.json --episodes 1000

# PPO training
python scripts/train.py --scenario cs_lite --config configs/cs_lite/cs_lite.json

# Full auto-curriculum (multi-phase PPO with auto-advancement)
python scripts/train.py --scenario cs_lite --mode curriculum --curriculum configs/cs_lite/curriculum.yaml

# Evaluate a trained model
python scripts/evaluate.py --model runs/<run>/best_model.zip --config configs/cs_lite/cs_lite.json --episodes 10

# Reward search (hyperparameter sweep)
python scripts/reward_search.py --config configs/cs_lite/cs_lite.json

# Watch a trained model play
python scripts/watch_model.py --model runs/<run>/best_model.zip --config configs/cs_lite/cs_lite.json

# Web app
cd web-app && pnpm dev              # dev server on :5173
cd web-app && pnpm build            # production build
cd web-app && npx tsc --noEmit      # type check
cd web-app && pnpm test             # run vitest
cd web-app && pnpm test:watch       # vitest watch mode
```

## Architecture

Cargo workspace with 4 crates:

- **`crates/engine`** -- Core simulation. Bevy ECS with Rapier2D physics. Modular system schedule using `EnginePhase` sets: `ClearBuffers -> AiDecisions -> PrePhysics -> PhysicsStep -> PostPhysics -> GameLogic -> StateTransitions -> Telemetry`. Entry point is `TickRunner` (direct via `::new()` or customisable via `EngineBuilder`).
- **`crates/telemetry`** -- Sink trait (`TelemetrySink`) with implementations: `WsSink` (broadcast channel -> WebSocket clients), `FileSink` (append JSONL), `BufferSink` (in-memory ring for Python).
- **`crates/server`** -- Axum HTTP/WS server. CLI via clap (`--port`). Spawns a `TickRunner` in a tokio task, exposes `/api/*` endpoints and `/ws/observe` + `/ws/play`. Registers itself in `~/.ghostlobby/sessions.json` on startup, cleans up on Ctrl-C.
- **`crates/py`** -- PyO3 module (`ghostlobby`). Exposes `GhostLobbyEnv` with `reset()`/`step()` for Gymnasium-style RL training. Supports multiple scenarios.

## Key Patterns

**Scenario system** -- Game modes implement the `Scenario` trait (8 methods: name, action_space, observation_space, setup, register_systems, observe, reward, is_done). Each scenario is a self-contained game definition. Current scenarios: `CsLiteScenario` (3D FPS with Rapier3D), `CsLiteDummyAiScenario` (CsLite with built-in dummy opponent), `TacticalDeathmatchScenario` (2D+A*).

**Action system** -- Actions are flat `Vec<f32>` arrays with multi-head definitions (`ActionSpaceDef`). Each head is either `Continuous { size, low, high }` or `Discrete { n }`. `RawActionBuffer` stores per-entity actions. `ActionMaskBuffer` provides per-tick validity masks.

**Observation system** -- Scenarios implement `observe()` which fills an `ObsWriter` with named feature arrays (self_features, entities, action_mask, etc.). `ObservationSpaceDef` describes shapes. `AgentRegistry` maps agent indices to ECS entities.

**Scripted AI** -- `ScriptedAi` component holds a closure (`AiFn`) that produces `ActionDict` from `AiContext`. Same action format as RL agents.

**Physics** -- `PhysicsState` wraps Rapier2D directly (not bevy_rapier). `Physics3DState` wraps Rapier3D for 3D scenarios (cs_lite). `PhysicsHandle`/`PhysicsHandle3D` components link ECS entities to Rapier rigid bodies/colliders.

**Config-driven** -- All game params live in JSON. Configs are organized by scenario: `configs/cs_lite/`, `configs/tactical/`. `GameConfig.extra` holds scenario-specific extensions.

**Telemetry** -- `TelemetryBuffer` resource collects events per tick. Events: `WorldSnapshot`, `Damage`, `Kill`, `Spawn`, `ShotFired`, `RoundStart`, `TickComplete`, `TacticalState`, `Arena3DState`.

## Session Auto-Discovery

Servers register themselves in `~/.ghostlobby/sessions.json` on startup. The web app auto-discovers active sessions via a Vite middleware at `GET /api/discover`.

**Registry format:** `[{ pid, port, title, config_path, scenario, started_at }]`

**Rust server** writes on startup, removes on Ctrl-C. Prunes dead PIDs on each register.

## Server Endpoints

- `GET /api/health` -- liveness
- `GET /api/match` -- current tick, title, status
- `POST /api/match/reset` -- reset simulation
- `GET /api/config` -- current GameConfig as JSON
- `GET /api/training` -- training status (stub in standalone server)
- `GET /api/obstacles` -- obstacle and spawn point layout
- `GET /ws/observe` -- read-only telemetry WebSocket
- `GET /ws/play` -- bidirectional WebSocket (send `{ "source_id": 0, "actions": [...] }`)

## Training Pipeline

Research-backed pipeline: **BC warm-start -> PPO fine-tune with KL anchor + entropy annealing**. Validated: 61% improvement over training from scratch at same step count.

### Python Packages (`python/`)

- **`glgym/`** -- Gymnasium wrappers. `BaseGhostLobbyGym` base class with `CsLiteGym`, `TacticalGym` subclasses. Handles obs flattening, action remapping (continuous -> discrete bins), curriculum phase masking, opponent management.
- **`training/`** -- Training infrastructure:
  - `bc_collector.py` -- Collect (obs, action) demonstrations from scripted AI via the native Rust collector.
  - `bc_pretrain.py` -- `BCTrainer`: PyTorch supervised learning on demos, saves SB3-compatible .zip + .pt reference for KL anchor.
  - `ppo_trainer.py` -- `PPOTrainer`: Unified PPO training with optional KL anchor, entropy schedule, self-play, plateau stopping.
  - `curriculum.py` -- `CurriculumRunner`: YAML-driven multi-phase training with auto-advancement when eval reward exceeds threshold.
  - `callbacks.py` -- SB3 callbacks: `KLAnchorCallback`, `EntropyScheduleCallback`, `PlateauStopCallback`, `SelfPlaySwapCallback`, `ThroughputCallback`.
  - `utils.py` -- `resolve_config`, `make_run_dir`, `make_vec_env`, `load_model`.
- **`scripts/`** -- CLI entry points: `train.py`, `collect_demos.py`, `evaluate.py`, `reward_search.py`, `watch_model.py`.

### Config Organization

```
configs/
  cs_lite/          # 3D FPS scenarios + reward configs
  tactical/         # 2D tactical scenarios
```

### Training Runs

Runs saved to `runs/{name}_{timestamp}/` with `experiment.json`, `checkpoints/`, `best/`, `eval_logs/`, `tb/` (TensorBoard).

## Web App (`web-app/`)

React 19 + TypeScript + Vite + Zustand + React Query + React Router.

### Pages

- **`/`** (HomePage) -- Dashboard showing all active sessions. Auto-discovers from `~/.ghostlobby/sessions.json` via `/api/discover` polling. Manual "Add Server" for remote servers.
- **`/viewer?host=localhost:3000`** (ViewerPage) -- Connects to a specific server. Uses the scenario registry to pick the right Canvas + sidebar panels. Shows live tick/TPS/entity count in header.

### Scenario Registry (`src/scenarios/`)

Each scenario type gets its own folder with a `ScenarioDefinition` export:

```typescript
interface ScenarioDefinition {
  id: string;
  name: string;
  match: (config: GameConfig) => boolean;  // first match wins
  Canvas: ComponentType;                    // main viewport
  sidebarPanels: ComponentType[];           // sidebar content
  onTelemetryEvent: (event: TelemetryEvent) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}
```

Register in `src/scenarios/registry.ts`. The `demo` scenario is the fallback (placeholder viewport).

**To add a new scenario viewer:** create `src/scenarios/<name>/`, export a `ScenarioDefinition`, register it in `registry.ts`.

### Key Files

- **`src/contexts/server.tsx`** -- `ServerHostProvider` / `useServerHost()` for per-viewer server targeting
- **`src/lib/server-url.ts`** -- `apiUrl(host, path)` / `wsUrl(host, path)` -- routes through Vite proxy in dev
- **`src/stores/dashboard-store.ts`** -- Manual server entries (localStorage)
- **`src/stores/viewer-store.ts`** -- Shared viewer state (connected, tick, tps, entityCount)
- **`src/hooks/use-discover.ts`** -- Polls `/api/discover` for auto-discovered sessions
- **`src/hooks/use-websocket.ts`** -- `useViewerWebSocket(host, scenario, enabled)` -- connects WS, dispatches to scenario
- **`src/hooks/use-game-config.ts`** -- React Query fetch of `/api/config`
- **`src/hooks/use-training-info.ts`** -- Polls `/api/training`

### Vite Config

- **Discovery plugin** -- serves `~/.ghostlobby/sessions.json` at `GET /api/discover`
- **Dynamic proxy** -- `/proxy/:port/*` routes to `http://localhost::port/*` (avoids CORS in dev)

### Testing

Vitest with jsdom, `@testing-library/react`, `@testing-library/jest-dom`. Tests live next to source files as `*.test.ts(x)`.

```bash
pnpm test          # single run
pnpm test:watch    # watch mode
```
