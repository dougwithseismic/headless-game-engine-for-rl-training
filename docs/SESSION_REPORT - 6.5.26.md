# GhostLobby Training Session Report

**Date:** 2026-05-05  
**Duration:** ~5 hours (17:00 - 22:00)  
**Total Training Steps:** 28M  
**Wall Clock Training Time:** ~49 minutes (across 4 curriculum phases)

---

## 1. Summary

We built a complete RL training pipeline for GhostLobby, a headless ECS game engine in Rust designed for agent training. Starting from a basic 1v1 deathmatch scenario that couldn't break past -126 reward (agents wandering aimlessly, dying to scripted AI), we iterated through 13 phases of architecture changes, reward shaping, observation design, and action space redesign. The final result: a curriculum-trained agent scoring +154 in open arenas and +112 with obstacles through 4-phase automated curriculum learning. Along the way we discovered that pre-normalized observations, continuous relative turn deltas, existence penalties, and staged curriculum learning were the critical differentiators between convergence and failure.

---

## 2. Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **Pre-normalized observations in the engine** | All obs values output in [-1, 1] from Rust. Eliminates VecNormalize, makes models portable, no stats drift between train/eval. |
| **Continuous relative turn delta** | Action outputs [-1, 1] as a fraction of max turn rate (8 rad/s). Network learns "turn left/right a bit" rather than computing absolute angles. 3x improvement over discrete bins. |
| **-1/+1 team encoding** | Symmetric around zero instead of 0/1. Prevents asymmetric network activation patterns that bias one team. |
| **Existence penalty (-0.0005/tick)** | Camping accumulates negative reward. Only offensive play can break even. Forces engagement. |
| **Round time limit (15s)** | Prevents infinite stalling. Timeout gives -0.5 penalty, teaching urgency. |
| **Line-of-sight raycasting** | Enemies behind walls are zeroed out in observations. Agent must search, not wallhack. Audio channel compensates for partial observability. |
| **Audio channel** | Shot bearing + proximity, works through walls. Gives agents directional information about unseen combat. |
| **Spawn point pool (8 positions)** | Randomly selected each round. Prevents policy from memorizing spawn-specific strategies. |
| **Obstacle jitter + rotation** | +/-40px positional noise, random 90-degree rotation each episode. Forces generalization over geometry. |

---

## 3. What Worked

- **Curriculum learning** -- Combat first (open arena, 3M steps), obstacles second (resume from combat model, 5M steps), then extended self-play (2x 10M steps). Each phase builds on converged skills from the previous phase.

- **Continuous relative turn delta** -- 3x improvement over discrete bins. The network directly outputs a steering signal rather than selecting from quantized buckets, eliminating overshoot and spinning behavior.

- **Self-play with checkpoint buffer** -- Pool of 20 past opponent snapshots, randomly sampled. Prevents the degenerate case where both agents converge to the same strategy and co-adapt into mutual incompetence.

- **Existence penalty** -- The single most impactful reward change. Without it, camping (never engaging) was weakly optimal. With it, passive play always accumulates cost.

- **Pre-normalized observations** -- Eliminated an entire class of bugs (VecNormalize state drift, train/eval mismatch, non-portable models). Every model checkpoint is self-contained and evaluates identically regardless of when it's loaded.

- **Warmup vs scripted AI before self-play** -- Agents learn basic combat against a predictable opponent before facing themselves. Critical for bootstrapping.

---

## 4. What Didn't Work

| Approach | Failure Mode |
|----------|-------------|
| **Throwing all complexity in at once** | Obstacles + random spawns + LOS simultaneously. Agent couldn't learn any single skill -- reward signal too noisy. |
| **Absolute angle action space** | Network had to learn coordinate systems/trigonometry before it could aim. Reward was too delayed from action. |
| **Discrete turn bins (7 bins)** | Quantization caused overshoot. Agent oscillated past targets, leading to spinning behavior at medium distances. |
| **VecNormalize** | Stats drift between training and evaluation. Models weren't portable -- needed `.pkl` files. Pre-normalized obs made it unnecessary. |
| **0/1 team encoding** | Asymmetric activation. Team 0 (value 0) gets zero gradient contribution from the team feature. Team 1 always active. Network learns team-specific policies. |
| **Fixed opponent self-play** | Both agents converge to same degenerate strategy (e.g., both camp because neither attacks first). No diversity pressure. |
| **Potential field pathfinding** | Local minima around obstacles. Scripted AI got permanently stuck on center pillar. |
| **Linear LR + clip decay** | Decayed too quickly (LR hit near-zero at 1.35M steps), freezing further improvement. Constant LR 3e-4 worked better. |

---

## 5. Training Results

### Final Curriculum Pipeline (28M steps, ~49 minutes)

| Phase | Config | Steps | Warmup | Start Reward | Peak Reward | Notes |
|-------|--------|-------|--------|--------------|-------------|-------|
| P1: Combat | `1v1_open.json` | 3M | 1M scripted | +17 | **+154** | Open arena, pure aim/shoot/dodge |
| P2: Obstacles | `1v1_deathmatch.json` | 5M | 500K scripted | +110 | **+122** | Resume from P1, add walls + LOS |
| P3: Self-play | `1v1_deathmatch.json` | 10M | 0 (pure SP) | +107 | **+119** | Held +110 avg against past selves |
| P4: Self-play Extended | `1v1_deathmatch.json` | 10M | 0 (pure SP) | +109 | **+119** | Held +112 avg, stable convergence |

### Historical Progression (earlier phases)

| Run | Steps | Best Reward | Key Change |
|-----|-------|-------------|------------|
| v2_rich_obs | 1M | -103 | Rich observations, pre-normalized |
| v2_tuned | 2M | -88 | Linear LR decay (too aggressive) |
| Clean self-play | 5M | -9.34 | Removed VecNormalize |
| 10M self-play | 10M | **+40.37** | First positive convergence, open arena |
| Full features v1 | 10M | +26.35 | LOS + audio + obstacles + turn rate |
| Final randomized | 10M | -8 (best) | Everything at once -- never positive |
| Research fixes | 10M | +22.68 | Discrete turn + existence penalty + checkpoint buffer |
| **Curriculum final** | 28M | **+154 / +112** | Staged learning with all fixes |

---

## 6. Key Hyperparameters

```
Algorithm:          PPO (Stable-Baselines3)
Policy:             MlpPolicy [64, 64]
Learning rate:      3e-4 (constant)
Batch size:         256
n_steps:            2048
n_epochs:           10
gamma:              0.99
ent_coef:           0.005
Normalization:      None (pre-normalized in engine)

Environment:
  Parallel envs:    16
  Frame skip:       4
  Max episode steps: 2048

Self-play:
  Swap interval:    250K steps
  Checkpoint buffer: 20 past opponents
  Scripted warmup:  Phase-dependent (0 - 1M steps)

Game Config:
  Arena:            600x600
  Tick rate:        64 Hz
  Max speed:        200 units/s
  Turn rate:        8.0 rad/s
  Damage:           34 (3-hit kill at 100 HP)
  Fire rate:        0.3s cooldown
  Range:            400 units
  Respawn delay:    0.5s
  Round time limit: 15s (timeout penalty: -0.5)
```

---

## 7. Observation Space (35 features)

Total flat vector: 11 + 10 + (max_entities-1)*10 + 2 + 2 = 35 features (for 1v1).

### Self Features (11)

| Index | Feature | Range | Notes |
|-------|---------|-------|-------|
| 0 | pos_x | [0, 1] | Normalized by arena width |
| 1 | pos_y | [0, 1] | Normalized by arena height |
| 2 | vel_x | [-1, 1] | Normalized by max_speed |
| 3 | vel_y | [-1, 1] | Normalized by max_speed |
| 4 | hp | [0, 1] | current/max |
| 5 | facing_sin | [-1, 1] | sin(facing_angle) |
| 6 | facing_cos | [-1, 1] | cos(facing_angle) |
| 7 | team | {-1, +1} | Symmetric encoding |
| 8 | weapon_cooldown | [0, 1] | Fraction of fire_rate remaining |
| 9 | nearest_enemy_dist | [0, 1] | Normalized by arena diagonal |
| 10 | nearest_enemy_bearing | [-1, 1] | Normalized by PI |

### Entity Features (10 per entity)

| Index | Feature | Range | Notes |
|-------|---------|-------|-------|
| 0 | dx | [-1, 1] | Delta x normalized by diagonal |
| 1 | dy | [-1, 1] | Delta y normalized by diagonal |
| 2 | distance | [0, 1] | Euclidean dist / diagonal |
| 3 | bearing | [-1, 1] | Relative angle / PI |
| 4 | vel_x | [-1, 1] | Enemy velocity normalized |
| 5 | vel_y | [-1, 1] | Enemy velocity normalized |
| 6 | hp | [0, 1] | Enemy health fraction |
| 7 | facing_sin | [-1, 1] | sin(enemy_facing) |
| 8 | facing_cos | [-1, 1] | cos(enemy_facing) |
| 9 | weapon_cooldown | [0, 1] | Enemy cooldown fraction |

**LOS gating:** If an enemy is not visible (blocked by obstacle), all 10 features are zeroed. The agent must use audio or memory to track hidden enemies.

### Audio (2)

| Index | Feature | Range | Notes |
|-------|---------|-------|-------|
| 0 | shot_bearing | [-1, 1] | Direction to most recent shot / PI |
| 1 | shot_proximity | [0, 1] | 1.0 = on top, 0.0 = max range |

Audio works through walls. Provides directional cues even when enemies are occluded.

### Action Mask (2)

| Index | Feature | Meaning |
|-------|---------|---------|
| 0 | can_act | 0 if dead, 1 if alive |
| 1 | can_shoot | 1 if weapon cooldown elapsed and alive |

---

## 8. Action Space (4 floats)

| Head | Type | Size | Range | Semantics |
|------|------|------|-------|-----------|
| move_dir | Continuous | 2 | [-1, 1] | 2D movement direction (x, y). Normalized internally, multiplied by max_speed. |
| turn | Continuous | 1 | [-1, 1] | Relative turn delta. Fraction of max turn rate (8 rad/s). +1 = full clockwise, -1 = full counter-clockwise. |
| shoot | Discrete | 2 | {0, 1} | Fire weapon (1) or don't (0). Gated by action mask. |

The continuous relative turn was the critical design choice. Previous iterations used:
- Absolute angle [-PI, PI]: required the network to learn trigonometry. Failed.
- Discrete 7-bin relative: quantization caused overshoot. Agents spun in circles.
- Continuous relative [-1, 1]: smooth steering signal. Converged 3x faster.

---

## 9. Infrastructure Built

### Training Tools

| File | Purpose |
|------|---------|
| `python/train_curriculum.py` | Auto-curriculum pipeline. Runs sequential phases, each resuming from the previous best model. Configurable phase list with per-phase configs, timesteps, and warmup durations. |
| `python/train_selfplay.py` | Core training script. PPO + self-play with checkpoint buffer, scripted warmup, eval callbacks, TensorBoard logging. |
| `python/selfplay_gym.py` | Gymnasium wrapper with self-play opponent management. Loads checkpoint buffer, swaps opponents on schedule. |
| `python/watch.py` | Model-vs-model or model-vs-scripted viewer. Loads two models, runs them in a shared env, streams telemetry to WebSocket for the web viewer. |
| `python/benchmark_throughput.py` | Measures raw engine throughput (ticks/sec) for performance regression testing. |

### Engine Features Added

| Feature | Description |
|---------|-------------|
| `RoundStart` telemetry event | Broadcasts obstacle positions + spawn points to clients on round reset. Enables dynamic obstacle rendering in the web viewer. |
| `/api/obstacles` endpoint | Returns current obstacle layout as JSON. Supports mid-round client joins that missed the `RoundStart` event. |
| Context-steering scripted AI | Random waypoint patrol + context steering for obstacle avoidance + pursuit mode on enemy visibility. Verified: 5 kills in 11 seconds. |
| Configurable round time limit | `spawning.round_time_limit` in config JSON. Triggers round reset with -0.5 timeout penalty. |
| Spawn point pool | 8 positions, randomly selected each round with +/-15px jitter. |
| Obstacle jitter + rotation | +/-40px positional noise, random 90-degree rotation per episode. |
| Line-of-sight raycasting | Rapier2D raycast from agent to each enemy. Occluded entities zeroed in observations. |
| Audio channel | Shot origin bearing + proximity. Persists through walls. |
| Turn rate system | `facing_system` rate-limits rotation to `turn_rate` rad/s. |

---

## 10. Next Steps

- **Symmetric -1/+1 team encoding validation** -- Train a full run confirming the symmetric encoding produces identical performance regardless of which team the agent is assigned to.

- **LSTM / RecurrentPPO** -- Current MLP policy has no memory. Enemies that duck behind cover are immediately forgotten. A recurrent policy could maintain a belief state about hidden enemy positions using audio cues.

- **2v2 mode** -- The architecture supports it (team encoding, multi-entity observations, variable agent count). Needs: teammate coordination reward, assist tracking, focus-fire incentives.

- **Ammo + reload mechanics** -- Add resource management as a skill dimension. Agents would need to manage engagement timing around reload windows.

- **Procedural obstacle generation** -- Replace hand-crafted obstacle configs with a procedural generator. Training on diverse layouts would produce more robust navigation policies.

- **Population-based training (PBT)** -- Maintain multiple agent lineages with different hyperparameters. Cross-evaluate and prune. Would likely find better hyperparameter schedules than our fixed values.

- **Reward function evolution** -- The current shaping (proximity + aim quality + existence penalty) was hand-tuned. Automated reward search could find better signal.

- **Multi-scenario curriculum** -- Train sequentially across DeathmatchScenario, MobaLaneScenario, and RacingScenario to produce a general game-playing agent.

---

## Appendix: Reward Function

```
reward = combat_reward + shaping

combat_reward:
  Kill:           +1.0
  Death:          -1.0
  Damage dealt:   +0.5 * (damage / max_hp)
  Damage taken:   -0.3 * (damage / max_hp)

shaping (per tick, if alive):
  Proximity:      +0.003 * (1 - dist/arena_diagonal)
  Aim quality:    +0.002 * max(0, dot(facing, to_enemy))
  Wall penalty:   -0.001 * wall_proximity_factor
  Existence:      -0.0005

timeout (round exceeds 15s):
  Timeout:        -0.5
```
