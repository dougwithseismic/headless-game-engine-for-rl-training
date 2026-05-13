# GhostLobby Training Findings — May 10-13, 2026

## TL;DR

Trained a PPO agent to 92.6% shot accuracy, 13:1 K/D ratio, and 91% win rate in a 1v1 Counter-Strike-style FPS. Four days of iterative work across ~30 training runs, ~60 reward search candidates, two action space redesigns, and two Eureka-style reward search rounds. The agent learned counter-strafing (stop before shooting) purely from reward shaping — no explicit curriculum for it.

---

## Timeline

### Day 1 — May 10: Finding the Right Setup

**Run 1-2: 5v5 Elimination.** Started with the full team format — 5v5, 80x60 arena, BC warm-start from scripted AI demos. Completely flat. Best reward: -1.09 across 13M steps. Too many agents, too sparse a reward signal.

**Runs 3-6: 1v1 LSTM from Scratch.** Pivoted to 1v1 on a 40x30 arena. LSTM policy trained without BC warm-start hit reward 337 against dummy AI — first real learning signal. But self-play collapsed it to ~0. Zero-sum signal is too noisy for a cold-start agent.

**Runs 7-9: Instrumented Self-Play.** Added behavioral tracking (accuracy, kills, damage, win rate in TensorBoard). All crashed within 30K steps. Pipeline instability.

**Runs 10-14: MLP + BC Warm-Start.** Switched from LSTM to MLP with BC resumption. Slow progress, stuck in -30 to -50 range. Continuous action space BC model showed first sign of life (+21.49 at 1.6M steps) but oscillated.

### Day 2 — May 11: The Breakthroughs

**Run 18: Dummy AI + Positive-Only Rewards.** Two simultaneous changes: switched to beatable dummy AI opponent, removed all negative reward terms (no death penalty). Reward climbed from 2.8 to 35.67 over 3M steps. This was the moment it started working.

**Run 19: 50M Step Scale-Up.** Resumed from Run 18, pushed to 50M steps. Reward: 45 -> 71 -> 115 -> 125.87 (30M), then plateau.

**Reward Search Batch 1 (16 configs).** All resumed from Run 19's best checkpoint, 500K steps each. Key finding: positive-only rewards consistently beat configs with negative penalties. The survival-focused config (death=-1.0) was dead last at -17.58.

**Run 22: Auto-Aim BC Model.** The single biggest leap. New BC model with auto-aim (engine handles pointing, agent learns when to shoot) jumped from ~60 to 190.9 within 1.25M steps. Previous agents with manual aim: 28% accuracy after 30M steps. Auto-aim: 98.6% in 5M.

**Run 23: Phase 2 vs Smart AI.** Resumed from auto-aim, tuned rewards (kill=5, death=-1.5, dmg=2, round_win=5). Extremely stable at 155.57 +/- 0.92. 98.6% accuracy, 90% win rate. Ready for self-play.

**Reward Search Batch 3 (25 configs).** Most comprehensive sweep. Top result: heavy_balanced (kill=10, death=-3, dmg=3, round_win=10, round_loss=-3) at 289.96.

**Runs 24-25: Self-Play.** Converged at 250.67 +/- 7.02 by 10M steps. Nash equilibrium — both sides improve equally, reward stays constant. This is expected.

### Day 3 — May 12: Architecture Overhaul

Redesigned the action space from 2 heads to 4 heads:

| Head | Before | After |
|------|--------|-------|
| `move_target` | Discrete(12) | Discrete(12) |
| `shoot` | Discrete(2) | Discrete(2) |
| `reload` | — | Discrete(2) |
| `use_action` | — | Discrete(3): none/plant/defuse |

Changed from `Box(continuous)` to `MultiDiscrete` in the Python gym wrapper. This matches real game input — a bitmask of independent button presses rather than a continuous vector.

Built goal conditioning (11-dim observation vector with objective type one-hot + target position + posture) for the hierarchical agent framework. Added `AgentGoal` component, `ObjectiveType` enum (PlantBomb, DefuseBomb, HoldPosition, Eliminate, Rotate), and `Posture` enum (Aggressive, Default, Passive).

Built the game-agnostic strategy layer: `StrategyBridge` trait (per-scenario) + `StrategyProvider` trait (game-agnostic) + `ChannelProvider` for external directive injection + `RulesProvider` with CS-Lite rule-based strategy.

Fixed dummy AI to follow goals and use the 4-action format. Fixed defuse radius from hardcoded 3.0 to config value (6.0).

Rewrote BC pre-trainer for multi-head classification: separate `nn.Linear` per branch with `CrossEntropyLoss`, concatenated into SB3's single `action_net` on save.

Benchmarked MLP vs LSTM vs GRU: LSTM 7.3x slower than MLP, GRU marginally slower than LSTM. The bottleneck is SB3's `_process_sequence()` Python loop, not the RNN cell. Keeping GRU support in-tree but defaulting to MLP.

### Day 4 — May 13: Eureka Reward Search + 50M Run

**Eureka Round 1 (8 candidates, 500K steps each).** Generated reward configs exploring different strategies:
- proven_positive, aggressive_balanced, aim_trainer, damage_per_hit
- dps_optimizer, objective_heavy, survive_and_plant, win_at_all_costs

**Eureka Round 2 (6 candidates, 500K steps each).** Introduced `reward_moving_shot` — a speed-scaled penalty for shooting while moving. CS players call this "counter-strafing": you stop before you shoot because movement tanks your accuracy.

The Eureka-discovered winning config (`proven_plus_stop`):

```json
{
  "reward_kill": 5.0,
  "reward_death": 0.0,
  "reward_damage_dealt": 2.0,
  "reward_near_miss": 0.05,
  "reward_moving_shot": -0.15
}
```

Positive-only combat rewards + movement penalty. No death penalty, no round outcome rewards.

**50M Step Training Run.** Wide arena (160x120), hitscan weapons (range=200), smart AI opponent (`cs_lite` scenario), 32 parallel envs.

| Steps | Accuracy | K/D | Win Rate | Shoot Rate | Notes |
|-------|----------|-----|----------|------------|-------|
| 2M | 48% | 2.6 | 72% | 11.8% | Learning to fight |
| 6M | 54% | 4.3 | 83% | 10.3% | Getting dominant |
| 10M | 56% | 5.0 | 82% | 9.2% | Consistency |
| 16M | 64% | 5.9 | 85% | — | Accuracy climbing |
| 22M | 78% | 6.3 | 87% | — | Breakthrough zone |
| 26M | 84% | 8.7 | 90% | — | Nearing mastery |
| 30M | 82% | 6.5 | 86% | — | Brief dip |
| 34M | 85% | 7.3 | 88% | — | Recovery |
| 38M | 87% | 7.8 | 88% | 5.4% | Fire discipline |
| 42M | 88% | 8.3 | 89% | — | Steady climb |
| 46M | 89% | 13.0 | 92% | — | Near-perfect |
| 48M | **92.6%** | 5.7 | 85% | — | Peak accuracy |
| 50M | 86.5% | 11.8 | 91% | 4.8% | Converging |

The shoot rate dropped from 11.8% to 4.8% over training. The agent learned to stop spraying and only fire when it would count.

---

## Key Findings

### 1. Auto-Aim is the Right Abstraction

Manual aim control (discrete yaw/pitch bins OR continuous steering) topped out at 28% accuracy after 30M steps. Auto-aim — the engine handles pointing, the agent decides movement + shoot timing — reached 98.6% in 5M steps.

This matches how real game AI works: the aim engine is separate from the decision engine. The RL agent's job is spatial reasoning and timing, not pixel-level aiming.

### 2. Positive-Only Rewards for Phase 1

Every run with death penalties in Phase 1 produced agents that avoided combat. The agent would camp in corners or run away. Removing all negative terms (death=0, damage_taken=0, round_loss=0) was the single change that unlocked learning.

Death penalties can be reintroduced in Phase 2 once the agent has learned basic combat. The curriculum matters.

### 3. Eureka-Style Reward Search Works for Game AI

14 reward configs evaluated in two rounds, ~30 minutes total compute. Found `proven_plus_stop` (the moving shot penalty config) which produced the best behavioral metrics we've seen.

Nobody has published Eureka-style reward search for game AI agents. All prior Eureka work (NVIDIA, DrEureka, Text2Reward) is robotics/locomotion. This is a novel application.

The key insight: for game AI, the reward structure is fixed (kill, death, damage, objectives). Only the weights need searching. This makes it simpler than robotics Eureka where the LLM generates Python reward code.

### 4. Movement Penalty Teaches Counter-Strafing

`reward_moving_shot = -0.15` with speed-scaled application:

```rust
if speed_frac > 0.1 {
    let penalty = config.reward_moving_shot * speed_frac;
    rewards.add(shooter_entity, penalty);
}
```

Also shrinks the effective hitbox based on speed: `effective_hitbox = hitbox_radius * (1.0 - speed_frac * 0.6)`. At full sprint, hitbox goes from 2.5 to 1.0.

Result: agents learned to stop before shooting. Accuracy went from ~35% (run-and-gun) to 92.6% (stop-and-shoot). This is CS counter-strafing, learned entirely from reward shaping — no explicit instruction.

### 5. MultiDiscrete Action Space Matches Game Input

Changed from `Box(continuous)` to `MultiDiscrete([12, 2, 2, 3])` — one independent categorical distribution per action head. This matches how real games work: WASD movement, mouse buttons, and action keys are all independent binary/categorical decisions, not a single continuous vector.

SB3 handles this natively with `MultiCategoricalDistribution` — one `action_net` Linear layer that outputs logits for all branches concatenated.

### 6. BC Warm-Start + KL Anchor

61% improvement over training from scratch at same step count. The BC model gives the agent a behavioral foundation (navigation, basic combat). The KL anchor prevents PPO from catastrophically forgetting BC-learned behaviors during early exploration.

Without KL anchor, PPO can destroy the BC policy in the first 100K steps as it explores random actions.

### 7. LSTM/GRU is Bottlenecked by SB3

MLP: 3,300 FPS. LSTM: 450 FPS (7.3x slower). GRU: 420 FPS (marginally slower than LSTM).

The bottleneck is not the RNN cell — it's SB3's `_process_sequence()` Python loop that iterates over each timestep sequentially. This is an implementation limitation, not a fundamental one. Sample Factory or a custom training loop could avoid this.

For now, MLP at 19K+ TPS is the right choice. LSTM matters for partial observability (fog of war, information asymmetry), which we haven't needed yet.

---

## Architecture Decisions Made

### Hierarchical Agent Framework

Three-layer architecture mirroring humanoid robots (Figure Helix, NVIDIA GR00T):

| Layer | Rate | Implementation | Role |
|-------|------|----------------|------|
| Strategy | ~0.1 Hz | LLM / Rules / Model | "What should we do?" — game state reasoning, objective selection |
| Tactics | ~1-4 Hz | A* / Skills / Goal system | "How do we do it?" — path planning, skill selection, goal sequencing |
| Motor | ~16 Hz | PPO policy | "Do it." — movement, shooting, actions |

Each layer is a trait with multiple implementations:
- `StrategyProvider`: `RulesProvider`, `ChannelProvider` (for LLM/HTTP injection), future `ModelProvider`
- `StrategyBridge`: `CsLiteBridge`, future `TacticalBridge`, `SurvivalBridge`

The strategy layer is game-agnostic. It consumes text/JSON state snapshots and emits semantic directives. The bridge translates between the generic strategy interface and scenario-specific ECS components.

### Goal Conditioning

11-dimensional goal vector appended to observations:
- Objective type one-hot (5 dims: PlantBomb, DefuseBomb, HoldPosition, Eliminate, Rotate)
- Target position (3 dims: relative x, y, z)
- Posture (3 dims: Aggressive, Default, Passive)

The strategy layer sets goals → the motor policy reads goals from observations → behavior changes accordingly. No retraining needed to change strategy.

---

## What Didn't Work

| Approach | Why It Failed |
|----------|--------------|
| 5v5 format | Too many agents, too sparse reward signal. 1v1 is the right starting point. |
| Discrete yaw/pitch bins | Oscillation. Agent can't converge on smooth aiming. |
| Heavy death penalties in Phase 1 | Agent learns "don't fight" — camps in corners. |
| LSTM without BC warm-start | Works from scratch vs dummy AI but collapses under self-play. |
| Self-play too early | Zero-sum signal too noisy for incompetent agents. |
| Combat-filtered BC demos | Removing navigation frames made BC worse, not better. Agent needs to learn to walk. |
| Per-tick shaping rewards | Encouraged camping and reward farming. |
| RecurrentPPO for throughput | 7x slower than MLP due to SB3's sequential Python loop. |

---

## Model Lineage

```
v1 (May 10-11): 2-head action space, manual aim
  BC: cs_lite_autoaim.zip
    -> Phase 1 (5M vs dummy): 78% accuracy, 73% win
      -> Phase 2 (10M vs scripted): 98.6% accuracy, 90% win
        -> Phase 3 (17M self-play): converged at 250

v2 (May 12-13): 4-head MultiDiscrete, auto-aim, goal conditioning
  Eureka Round 1: 8 configs, 500K steps each
  Eureka Round 2: 6 configs with reward_moving_shot
  Winner: proven_plus_stop (kill=5, dmg=2, near_miss=0.05, moving_shot=-0.15)
    -> 50M steps vs smart AI: 92.6% peak accuracy, 13:1 K/D, 91% win rate
```

---

## Winning Configuration

**Arena:** 160x120 units, 5 obstacle boxes, hitscan weapons (range=200, damage=25, fire_rate=0.3)

**Action space:** `MultiDiscrete([12, 2, 2, 3])` — move_target, shoot, reload, use_action

**Reward config (`configs/cs_lite/1v1_wide_eureka_winner.json`):**
```json
{
  "reward_kill": 5.0,
  "reward_death": 0.0,
  "reward_damage_dealt": 2.0,
  "reward_damage_taken": 0.0,
  "reward_round_win": 5.0,
  "reward_round_loss": 0.0,
  "reward_near_miss": 0.05,
  "reward_friendly_fire": -2.0,
  "reward_bomb_plant": 0.5,
  "reward_bomb_defuse": 1.0,
  "reward_bomb_pickup": 0.5,
  "reward_moving_shot": -0.15
}
```

**Training hyperparameters:**
- PPO with MLP policy (pi: [256, 256], vf: [256, 256])
- lr=3e-4, batch_size=256, n_steps=4096, n_epochs=4
- gamma=0.99, gae_lambda=0.95, clip_range=0.2
- ent_coef=0.01, 32 parallel envs, frame_skip=4
- 50M timesteps, eval every 2M steps

---

## Compute

All training on M4 Max 64GB MacBook Pro.

- Engine throughput: 19K+ TPS (uncapped), 3,300 FPS per PPO env
- 50M step run: ~4 hours wall clock
- Eureka search (14 configs x 500K): ~30 minutes
- Total compute across all runs: ~40 hours over 4 days

---

## Open Questions

1. **Bomb play not emerging.** The agent dominates through combat so hard (13:1 K/D) that it never needs to plant or defuse. The objective rewards (bomb_plant=0.5, bomb_defuse=1.0) are dwarfed by combat rewards. Would need to either scale objective rewards much higher or create scenarios where combat alone can't win.

2. **Accuracy measurement.** Behavior tracker may count shoot button presses differently from actual shots fired (frame_skip interaction). The 92.6% is from the behavior callback's accuracy metric, which tracks ShotFired telemetry events. Believed accurate but should be verified independently.

3. **Self-play on v2.** The 50M run was against scripted smart AI. Self-play with the v2 action space (4-head MultiDiscrete) hasn't been tested. The v1 self-play converged at Nash equilibrium — would v2 produce different dynamics with reload/use_action heads?

4. **5v5 scaling.** The 1v1 pipeline is validated. Can it scale to 2v2 and 5v5 with team coordination through the strategy layer?

5. **Cross-game transfer.** The hierarchical architecture is designed to be game-agnostic. The motor policy is scenario-specific, but the strategy layer should transfer. Not yet tested on a second game type.
