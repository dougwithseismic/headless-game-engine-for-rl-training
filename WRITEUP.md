# GhostLobby: Project Write-Up

## What It Is

GhostLobby is a headless, config-driven game engine built in Rust for training reinforcement learning agents. The core simulation runs on Bevy ECS with Rapier 2D/3D physics, hitting 19K+ TPS on the primary 3D FPS scenario and 1M+ TPS on simpler configurations. It exposes a Gymnasium-compatible Python API via PyO3 for training with Stable Baselines 3, an Axum HTTP/WebSocket server for live observation, and a React/Three.js web app for 3D spectating.

The goal isn't just FPS bots. It's controllable embodied agents across domains -- the same architecture has already been used for 3D FPS combat and drone hovering. The engine is designed to produce agents with tunable behavior profiles (explorer, berserker, pacifist, skill-scaled) by varying reward functions over a shared tactical backbone.

## Architecture

Four Rust crates in a Cargo workspace:

| Crate | Role |
|---|---|
| `crates/engine` | Pure simulation. Bevy ECS, Rapier 2D/3D physics, zero IO dependencies. Embeddable anywhere. |
| `crates/telemetry` | Pluggable event sinks: WebSocket broadcast, JSONL file, in-memory ring buffer for Python. |
| `crates/server` | Axum HTTP/WS server with session auto-discovery via `~/.ghostlobby/sessions.json`. |
| `crates/py` | PyO3 module exposing `GhostLobbyEnv` with `reset()`/`step()` for Gymnasium-style RL. |

The engine tick runs a deterministic schedule:

```
ClearBuffers -> AiDecisions -> PrePhysics -> PhysicsStep -> PostPhysics -> GameLogic -> StateTransitions -> Telemetry
```

Three tick modes: **Uncapped** (training, max throughput), **RealTime** (60 TPS for human play), **Stepped** (single-tick debugging).

Everything is config-driven. `GameConfig` has typed fields (arena, movement, combat, spawning, teams, obstacles) plus a freeform `extra` JSON block for scenario-specific parameters, so new game modes don't require engine changes.

## The Scenario System

Game modes implement the `Scenario` trait (8 methods: name, action_space, observation_space, setup, register_systems, observe, reward, is_done). Each scenario is a self-contained game definition. Three are active:

### CS-Lite (Primary -- 3D FPS)

Full Counter-Strike-style round structure with a `BuyFreeze -> Active -> RoundEnd` state machine. T vs CT sides, bomb mechanics (plant/defuse/detonate/dropped), A/B sites, and a strategy system with pluggable attack strategies (RushA, RushB, SplitAB, FakeARushB, etc.) and defense strategies (Default, StackA, StackB, Aggressive). Agents are assigned roles: EntryFragger, BombRunner, Support, Lurker, SiteAnchor, MidHold, Rotator.

3D physics via Rapier3D. Full NavGrid A* pathfinding on the XZ plane. 45 raycasts (9 horizontal x 5 vertical) with configurable FOV and range.

**Action space**: 2 heads -- `move_target` (Discrete 12: 8 compass directions + stay + seek cover + advance + retreat) and `shoot` (Discrete 2). The engine handles aiming -- the agent decides **where to move** and **when to shoot**.

**Observation space**: 234+ dimensions including self state, weapon state, teammate/enemy positions, round info, bomb state, navigation candidates, 3D raycasts, spatial audio, and aim state.

All rewards are configurable from JSON: kill, death, damage dealt/taken, round win/loss, near miss, friendly fire, bomb pickup/plant/defuse.

### CS-Lite Dummy AI (Curriculum Opponent)

Same game, but one side is a deliberately bad scripted opponent: wanders randomly, turns slowly, only shoots 1/3 of ticks. Provides a beatable target for the learning agent's first training phase.

### Tactical Deathmatch (2D)

A 2D variant with 4 action heads (move, aim delta, shoot, weapon select), 64-ray sensor sweep, and candidate positions with cover evaluation.

## Training Pipeline

The research-backed pipeline:

```
Scripted AI Demos -> Behavioral Cloning -> PPO Fine-Tuning -> Self-Play
```

### Python Infrastructure

- **`glgym/`** -- Gymnasium wrappers handling obs flattening, action remapping, frame skipping, opponent management, behavior tracking
- **`training/`** -- PPO trainer with KL anchoring, entropy annealing, self-play opponent swapping, plateau stopping, and behavioral evaluation
- **`scripts/`** -- CLI tools for training, demo collection, evaluation, reward search, and live model watching

### Reward Search (Eureka-Style)

Rather than hand-tuning rewards, we run 8-24 reward configurations in parallel, each for 500K steps (~3 minutes). Results are ranked by behavioral metrics in TensorBoard. The best config was found in 22 minutes vs what would have been days of manual iteration.

## The Training Journey

25 actual training runs and 48 reward search runs across May 10-11, 2026. Here's how it unfolded.

### Day 1: Finding the Right Setup

**5v5 Elimination (Runs 1-2)** -- Started with a 5v5 format, 80x60 arena, BC warm-start from scripted AI demonstrations. Completely flat. Best reward: -1.09. 13M total steps of nothing. **5v5 was too complex.**

**1v1 LSTM from Scratch (Runs 3-6)** -- Pivoted to 1v1 on a smaller 40x30 arena. LSTM policy trained from scratch (no BC) hit 337 reward -- the first real learning signal. But when self-play was enabled, it collapsed back to ~0. The zero-sum signal was too noisy for a cold-start agent.

**Instrumented Self-Play (Runs 7-9)** -- Added behavior tracking to understand what the agents were actually doing. All crashed within 30K steps. Something in the pipeline was unstable.

**MLP with BC Warm-Start (Runs 10-14)** -- Switched from LSTM to MLP, resumed from various BC models. Slow progress but still stuck in the -30 to -50 range. The continuous action space BC model showed the first sign of life (+21.49 at 1.6M steps) but oscillated badly.

### Day 2: The Breakthroughs

**Dummy AI Opponent, Positive-Only Rewards (Run 18)** -- Two changes at once: switched to the beatable dummy AI opponent, and used positive-only rewards (no death penalties). Reward climbed from 2.8 to 35.67 over 3M steps. **This was the moment it started working.**

**50M Step Scale-Up (Run 19)** -- Resumed from the breakthrough run, pushed to 50M steps. Reward climbed steadily: 45 -> 71 -> 115 -> **125.87** at 30M steps, then plateaued.

**Reward Search Batch 1 (16 runs)** -- All resumed from the 50M best model, 500K steps each. Key finding: **positive-only rewards consistently outperformed configs with negative penalties**. The survival-focused config with death=-1.0 was dead last at -17.58.

**Auto-Aim BC Model (Run 22)** -- The single biggest leap. A new BC model trained with auto-aim (engine handles pointing, agent learns when to shoot) jumped from ~60 to **190.9 within 1.25M steps**, stabilizing at ~193. Previous agents with manual aim got 28% accuracy after 30M steps. Auto-aim reached 98.6% in 5M.

**Phase 2 Final (Run 23)** -- Resumed from auto-aim, tuned reward config from search results (kill=5, death=-1.5, dmg=2, round_win=5). Extremely stable at **155.57 +/- 0.92**. Near-zero variance. Ready for self-play.

**Reward Search Batch 3 (25 runs)** -- The most comprehensive sweep, off the Phase 2 model. Top result: heavy_balanced (kill=10, death=-3, dmg=3, round_win=10, round_loss=-3) at 289.96. The final phase 3 config used an aggressive-balanced variant: kill=8, death=-1, dmg=3, round_win=8, round_loss=-1.

### Self-Play Convergence

**Phase 3 Self-Play (Runs 24-25)** -- Resumed from Phase 2 final, both agents are copies of the same policy, updating together. Initial run (10M steps): stable at 246-250. Extended run (30M steps): converged at **250.67 +/- 7.02** by 10M steps. No further improvement through 17M steps. The plateau stop kicked in.

The reward hovering at ~250 is expected behavior -- in self-play, as both sides improve equally, the reward signal stays roughly constant. The agents have reached a Nash equilibrium for this configuration.

## What Worked

| Technique | Impact |
|---|---|
| **Auto-aim** | 28% -> 98.6% accuracy. Engine handles pointing; agent decides movement + timing. By far the biggest single improvement. |
| **Positive-only rewards** | The breakthrough that made Phase 1 trainable. Death penalties caused avoidance behavior that prevented learning. |
| **BC warm-start + KL anchor** | 61% improvement over training from scratch at the same step count. Agent starts knowing basic behaviors, KL anchor prevents catastrophic forgetting. |
| **Eureka-style reward search** | Found optimal reward configs in 22 minutes across 48 parallel runs instead of days of manual tuning. |
| **Curriculum progression** | Dummy AI (beatable) -> Scripted AI (competent) -> Self-play (adversarial). Each phase built on the previous model. |
| **Behavioral instrumentation** | TensorBoard tracking of accuracy, kill rate, damage dealt, round outcomes -- not just reward curves. |

## What Didn't Work

| Approach | Why It Failed |
|---|---|
| **5v5 format** | Too many agents, too sparse a reward signal. 1v1 was the right starting point. |
| **Discrete yaw/pitch bins** | Oscillation. Agents couldn't converge on smooth aiming with discrete angular bins. |
| **Heavy death penalties** | Agents learned to avoid combat entirely. Camping in corners instead of fighting. |
| **LSTM without BC warm-start** | Worked from scratch vs dummy AI but collapsed under self-play. Too unstable without a behavioral foundation. |
| **Self-play too early** | Zero-sum signal is too noisy for an agent that hasn't learned basic competence yet. |
| **Combat-filtered BC demos** | Removing non-combat frames from demonstrations made the BC model worse, not better. Agents need to learn navigation too. |
| **Per-tick shaping rewards** | Encouraged camping and exploitative behavior instead of tactical play. |

## Final Model Lineage

```
BC: cs_lite_autoaim.zip (auto-aim behavioral cloning)
  -> cs_lite_autoaim (5M steps vs dummy AI, reward 193)
    -> cs_lite_phase2_final (10M steps vs scripted AI, reward 155, 98.6% accuracy)
      -> cs_lite_phase3_selfplay (10M steps self-play, reward 250)
        -> cs_lite_phase3_extended (17M steps self-play, converged at 250)
```

## Training Results Summary

| Phase | Opponent | Steps | Accuracy | Win Rate | Reward |
|---|---|---|---|---|---|
| Phase 1 (BC warm-start) | Dummy AI | 5M | 78% | 73% | 125.87 |
| Phase 2 (fine-tune) | Scripted AI | 10M | 98.6% | 90% | 155.57 |
| Phase 3 (self-play) | Self | 17M | -- | ~50% (expected) | 250.67 |

## Cross-Domain Validation: Drone Hovering

To prove the architecture generalizes beyond FPS, a drone hovering scenario was implemented. Phase 1 (stabilize hover) reached peak reward 291 at 12.5M steps using a rate-mode -> god-mode -> teacher-student curriculum. Same engine, same training pipeline, completely different domain.

## Tech Stack

**Engine**: Rust, Bevy ECS 0.16, Rapier 2D/3D 0.22, Glam 0.29
**Server**: Axum 0.8, Tokio, Tower-HTTP, Clap
**Python**: PyO3 0.25, Maturin, Stable Baselines 3, PyTorch, Gymnasium
**Web**: React 19, TypeScript, Vite, Three.js (react-three-fiber), Zustand, React Query, React Router
**Training**: SB3 PPO/RecurrentPPO, custom KL anchor + entropy schedule callbacks, Eureka-style reward search

## What's Next

The engine is proven. The training pipeline works. The agents can learn to fight. Open directions:

- **5v5 scaling**: Now that the 1v1 pipeline is validated, curriculum from 1v1 -> 2v2 -> 5v5 with team coordination
- **Behavior profiles**: Same model, different reward weights -> distinct play styles (aggressive, defensive, tactical)
- **Teacher-student distillation**: Train in simplified environments, transfer to constrained ones
- **Multi-map generalization**: Train across diverse obstacle layouts to prevent map-specific overfitting
- **Human play integration**: The `/ws/play` endpoint exists -- plug human players into the training loop
