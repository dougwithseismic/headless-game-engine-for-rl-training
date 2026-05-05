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

# Web app (React 19 viewer)
cd web-app && pnpm dev              # dev server on :5173, proxies to :3000
cd web-app && pnpm build            # production build
cd web-app && npx tsc --noEmit      # type check
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

Two viewer implementations exist:

**`web/index.html`** -- Original standalone Canvas 2D viewer. No build step, served directly by the Rust server at `http://localhost:3000`.

**`web-app/`** -- React 19 + TypeScript app (Vite, zustand, React Query). Full port of the original viewer with proper component architecture.

```bash
# Dev server (proxies /api/* and /ws/* to Rust server on :3000)
cd web-app && pnpm dev    # http://localhost:5173

# Production build
cd web-app && pnpm build  # outputs to web-app/dist/

# Type check
cd web-app && npx tsc --noEmit
```

### React App Architecture

The app splits **React-rendered UI** from the **imperative Canvas draw loop**:

- **React components** (re-render on state change): Header, Sidebar panels (Scoreboard, AgentList, KillFeed, Terminal), HUD overlays, FollowBanner, ZoomIndicator, EntityTooltip
- **Canvas draw loop** (60fps via rAF, reads zustand via `getState()`): All canvas rendering -- entities, effects, fog, minimap. Lives in `src/renderer/` as pure functions, not React components.
- **Mutable refs** (never trigger re-renders): particles, shotTraces, dmgNumbers, decals, ripples, prevPositions. Stored in an `EffectsState` ref passed between hooks.

### Zustand Stores (`src/stores/`)

- **game-store** -- entities, tick, entityIdMap, kills (max 12), score[], eventLog (max 200), tps, connected. Updated by WebSocket events.
- **camera-store** -- camX/Y, camZoom (1.0-8.0), followId, isPanning, shakeX/Y/Decay. Read by draw loop via `getState()`, written by camera controls hook.
- **render-store** -- fog/glow/grid/trails booleans. Toggled by CanvasControls buttons.

### Key Hooks (`src/hooks/`)

- **use-websocket** -- Connects to `/ws/observe`, dispatches events to zustand stores and effects ref. WorldSnapshot updates are rAF-gated to throttle React re-renders.
- **use-canvas-renderer** -- Owns the `requestAnimationFrame` draw loop. Calls renderer functions sequentially. Reads stores non-reactively.
- **use-camera-controls** -- Wheel/click/drag/keyboard handlers for zoom, pan, follow, and shortcuts (1-9 to select agents, Esc to reset).
- **use-game-config** -- React Query fetch of `/api/config` with `staleTime: Infinity`.

### Renderer (`src/renderer/`)

Pure imperative Canvas2D functions, each taking `ctx`, camera params, and data. Not React components. Pipeline order: background → arena bounds → grid → obstacles → decals → ambient → ripples → shots → particles → entities → damage numbers → fog → minimap.
