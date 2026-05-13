# GhostLobby

A headless game engine for training FPS agents using reinforcement learning. Built in Rust for speed (19K+ TPS), with Python bindings for PPO training via Stable Baselines 3.

The methodology combines behavioral cloning from a scripted AI teacher, curriculum training against progressively harder opponents, and automated reward function search inspired by NVIDIA's Eureka framework.

## What This Does

The engine simulates a 1v1 tactical shooter (CS-Lite) with 3D physics, A* navigation, cover mechanics, and raycasted combat. An RL agent learns to aim, move, shoot, and use cover by playing thousands of matches per minute against scripted opponents.

**Current results (Phase 1, 30M steps):**
- 28% shot accuracy (from 0% at step 0)
- 12+ kills per match against a basic opponent
- Continuous aim steering (no discrete bin oscillation)
- Full behavioral instrumentation tracking every metric in TensorBoard

## Training Methodology

### The Pipeline

```
Scripted AI demos → Behavioral Cloning → PPO Fine-tuning → Curriculum Progression
```

1. **Scripted AI** plays the game. We record every (observation, action) pair.
2. **Behavioral Cloning (BC)** trains a neural network to imitate the scripted AI via supervised learning. The agent starts knowing how to aim and shoot.
3. **PPO** refines the BC policy through reinforcement learning, improving beyond the teacher.
4. **Curriculum** increases opponent difficulty as the agent improves. Each phase has its own reward configuration.

### Curriculum Phases

| Phase | Opponent | What the agent learns | Gate to next phase |
|-------|----------|----------------------|-------------------|
| 1 | Dummy AI (slow aim, random movement) | Aim, shoot, basic combat | >80% win rate |
| 2 | Scripted AI (80% accuracy, flanking) | Survive against competent play | >50% win rate |
| 3 | Self-play (copies of itself) | Adapt, counter-strategy | Elo plateau |

### Reward Search (Eureka-style)

Instead of manually tuning reward functions, we test multiple configurations in parallel and compare behavioral metrics. Each candidate runs for 500K steps (~3 minutes), and the results are ranked automatically.

```bash
python scripts/reward_search.py \
  --base-config configs/cs_lite/1v1_tactical.json \
  --reward-dir configs/cs_lite/rewards_phase2/ \
  --scenario cs_lite \
  --steps 500000 \
  --bc-model data/bc_models/model.zip
```

Output:
```
Candidate          Accuracy  Kills/ep  Deaths/ep  Reward
trade               26.65%      1.9      13.5     +50.3  <- best accuracy
mild_penalty         25.36%      2.5      13.5     +28.1  <- best kills
aggressive           23.98%      1.8      13.5     +63.9
positive_only        22.55%      1.3      13.3     +18.5
```

Reward values are configured in JSON — no Rust recompilation needed:
```json
{
  "reward_kill": 5.0,
  "reward_death": -0.5,
  "reward_damage_dealt": 2.0,
  "reward_damage_taken": -0.1,
  "reward_round_win": 5.0,
  "reward_round_loss": -0.5,
  "reward_near_miss": 0.05
}
```

### Training Instrumentation

Every training run emits behavioral metrics to TensorBoard alongside standard PPO metrics. This catches degenerate behavior (camping, spinning, not shooting) within minutes instead of after hours of wasted compute.

**Behavioral metrics (`behavior/`):**
- `accuracy` — shot hit rate. 0% = can't aim. First thing to check.
- `kills_per_ep`, `deaths_per_ep` — is the agent fighting?
- `shoot_rate` — 0% = not shooting, >80% = spray-and-pray
- `damage_dealt_per_ep`, `damage_taken_per_ep` — combat intensity

**Reward components (`reward_components/`):**
- Per-category breakdown: `kill`, `death`, `damage_dealt`, `damage_taken`, `near_miss`, `round_win`, `round_loss`
- Shows exactly where reward comes from. If 100% comes from one source, something is wrong.

## Quick Start

```bash
# Build
cargo build
cargo test -p ghostlobby-engine

# Python setup
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy gymnasium stable-baselines3 sb3-contrib torch
cd crates/py && maturin develop --release && cd ../..

# Collect demos from scripted AI
cd python
python scripts/collect_demos.py \
  --scenario cs_lite_dummy \
  --config configs/cs_lite/1v1_tactical.json \
  --episodes 500

# BC pre-train
python scripts/train.py --scenario cs_lite_dummy --mode bc \
  --demos data/demos/cs_lite_dummy.npz \
  --config configs/cs_lite/1v1_tactical.json

# PPO training
python scripts/train.py --scenario cs_lite_dummy \
  --config configs/cs_lite/1v1_tactical.json \
  --timesteps 3000000 --n-envs 32 \
  --resume data/bc_models/cs_lite_dummy.zip \
  --kl-anchor data/bc_models/cs_lite_dummy_ref.pt

# Monitor
tensorboard --logdir runs/

# Reward search
python scripts/reward_search.py \
  --base-config configs/cs_lite/1v1_tactical.json \
  --reward-dir configs/cs_lite/rewards_phase2/ \
  --scenario cs_lite \
  --steps 500000 \
  --bc-model runs/best_model.zip

# Watch model play (web viewer must be running)
python scripts/watch_model.py \
  --config configs/cs_lite/1v1_tactical.json \
  --model runs/best_model.zip \
  --port 3000

# Web viewer
cd web-app && pnpm dev    # http://localhost:5173
```

## Architecture

```
crates/
  engine/       Bevy ECS simulation, Rapier 3D physics, scenarios
  server/       Axum HTTP/WebSocket server for web viewer
  telemetry/    Event sinks (WebSocket, JSONL, in-memory)
  py/           PyO3 bindings (GhostLobbyEnv)

python/
  glgym/        Gymnasium wrappers (CsLiteGym, TacticalGym)
  training/     BC collector, PPO trainer, behavioral callbacks
  scripts/      CLI: train.py, collect_demos.py, reward_search.py, watch_model.py

configs/
  cs_lite/      Game configs + reward search variants
    rewards/         Phase 1 reward configs
    rewards_phase2/  Phase 2 reward configs
```

### Engine Tick Schedule

```
ClearBuffers → AiDecisions → PrePhysics → PhysicsStep → PostPhysics → GameLogic → Telemetry
```

### Action Space (4 heads)

| Head | Type | Range | Purpose |
|------|------|-------|---------|
| `move_target` | Discrete(12) | 0-11 | 8 compass + stay + cover + advance + retreat |
| `yaw_delta` | Continuous | [-1, 1] | Fraction of max turn rate (smooth steering) |
| `pitch_delta` | Continuous | [-1, 1] | Fraction of max pitch rate |
| `shoot` | Discrete(2) | 0-1 | Fire weapon |

Yaw and pitch use **continuous relative steering** — the network outputs a fraction of the max turn rate, enabling smooth aim tracking without discrete bin oscillation.

### Observation Space (229 dims for 1v1)

Self state (12), weapon state (20), teammate state, enemy state (10), round info (9), bomb state (8), A* candidate positions (60), 3D raycasts (90), audio (6), action mask (14).

Audio includes gunshot bearing + freshness flag and footstep bearing + loudness from enemy movement, giving temporal signals for LSTM-based architectures.

### Scripted AI Variants

- **Smart AI** (`cs_lite`) — 80% accuracy, continuous aim tracking, flanking, peeking. The Phase 2+ opponent and the BC demo teacher.
- **Dummy AI** (`cs_lite_dummy`) — 40% turn speed, wide shoot threshold, random movement. The Phase 1 training opponent.

## Key Learnings

Things that worked:
- **Positive-only rewards for Phase 1** — no death penalty means PPO can't learn to avoid combat
- **Continuous aim** instead of discrete bins — eliminates overshoot oscillation
- **BC warm-start with KL anchor** — agent starts competent, PPO improves without destroying aim
- **Behavioral instrumentation** — catches broken training in minutes, not hours
- **Automated reward search** — 8 configs in 22 minutes vs days of manual tuning

Things that didn't work:
- **Discrete yaw/pitch bins** — perpetual oscillation, agent spins
- **Heavy death penalties** — PPO learns "don't fight" instead of "fight better"
- **Per-tick shaping rewards** — agent camps near cover collecting free reward
- **LSTM without BC warm-start** — random LSTM weights can't discover aim from scratch
- **Self-play too early** — two bad agents reinforcing each other's bad habits

## Recurrent Memory Benchmarks

For strategic game AI (CS2, AoE2), agents need cross-round memory — tracking bomb locations, enemy scouting patterns, economy state across hundreds of timesteps. We benchmarked LSTM and GRU memory architectures against the MLP baseline to understand the cost.

### Results

Benchmarked on cs_lite (3D FPS, 32 parallel envs, obs=229, act=4 heads) and a zero-cost synthetic env (obs=64, act=8) to isolate training loop overhead:

| Architecture | cs_lite (real env) | Synthetic (zero-cost env) |
|---|---|---|
| MLP | 13,700 steps/sec | 31,922 steps/sec |
| LSTM (hidden=256) | 1,833 steps/sec (7.5x slower) | 686 steps/sec (46x slower) |
| GRU (hidden=256) | 1,763 steps/sec (7.8x slower) | 750 steps/sec (43x slower) |

GRU has ~30% fewer parameters than LSTM (2 gates vs 3), but performs within noise of LSTM in practice. On the synthetic env where env cost is zero, the slowdown balloons to 40-46x — confirming the bottleneck is entirely in the training loop, not the RNN cell.

### Why GRU doesn't help

The bottleneck is sb3-contrib's `RecurrentActorCriticPolicy._process_sequence()`, which processes rollout sequences step-by-step in a Python loop. Both LSTM and GRU hit the same wall: sequential dispatch overhead dominates, making the gate count irrelevant.

### What this means for training

| Run type | MLP | LSTM/GRU |
|---|---|---|
| 10M step phase | 12 min | ~89 min |
| 30M self-play run | 36 min | ~4.4 hours |
| 45M+ full curriculum | ~1 hour | ~7 hours |

7 hours for a full curriculum is overnight territory — still very runnable. But iteration speed during research (reward tuning, obs space design) takes a real hit.

### Reproducing

```bash
cd python
python scripts/bench_memory.py
```

### Implications

If recurrent memory at speed becomes necessary (e.g., long self-play population training), the path forward is Sample Factory — its async architecture with PackedSequence BPTT addresses exactly the sequential rollout bottleneck that sb3-contrib can't avoid. The `--memory` flag supports `none`, `lstm`, and `gru` so architecture can be A/B tested without code changes:

```bash
python scripts/train.py --scenario cs_lite --config configs/cs_lite/cs_lite.json --memory gru --memory-hidden-size 512
```

## Future Direction

See [docs/eureka-direction.md](docs/eureka-direction.md) for the full plan to automate reward function design using LLM-guided search, following NVIDIA Eureka's framework applied to FPS game agents.

## Dependencies

**Rust:** bevy_ecs 0.16, rapier2d 0.22, rapier3d 0.22, glam 0.29, axum 0.8, tokio, serde, pyo3 0.25

**Python:** gymnasium, stable-baselines3, sb3-contrib, numpy, torch, maturin
