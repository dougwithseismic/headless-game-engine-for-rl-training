# GhostLobby Training Log

Chronological record of RL training experiments for the GhostLobby headless FPS engine.

---

## 2026-05-05

### 17:00 -- Phase 1: Foundation (DeathmatchScenario Baseline)

Started with a minimal deathmatch setup to establish a working training loop.

**Observation space:**
- 7 self-features: position (x, y), velocity (vx, vy), hp, facing angle, team
- 6 entity features per visible entity: dx, dy, vel_x, vel_y, hp, team

**Action space:**
- Continuous: move_dir (2 floats), look_angle (1 float)
- Discrete: shoot (2 choices -- fire or don't)

**Reward shaping:**
| Signal | Value |
|--------|-------|
| Kill | +1.0 |
| Death | -1.0 |
| Damage dealt | +0.5 * dmg/max_hp |
| Damage taken | -0.3 * dmg/max_hp |
| Proximity to enemy | +0.001 per tick |
| Aim quality | +0.0005 per tick |

**Config:** 25 damage per shot, 0.3s fire rate, 100 HP, 2s respawn delay, open 600x600 arena.

No quantitative results from this phase -- it was scaffolding to confirm the pipeline worked end-to-end.

---

### 17:19 -- Phase 2: Rich Observations + Hyperparameter Tuning

Enriched observations significantly and tuned the reward function to encourage engagement.

**Changes:**
- Expanded to 11 self-features + 10 entity features per entity
- Added: weapon cooldown timer, bearing to enemy, distance, sin/cos of facing angle
- Pre-normalized all observations to [-1, 1] range inside the Rust engine
- Bumped proximity reward 3x (0.001 -> 0.003)
- Bumped aim reward 4x (0.0005 -> 0.002)
- Lowered TTK: damage 25 -> 34 (3-hit kill instead of 4), respawn delay 2s -> 0.5s
- Added round-reset respawn: both agents teleported to symmetric positions on any kill

**Run: "v2_rich_obs" (1M steps)**
- Reward: -126 -> -103 (best) -> regressed back to -126
- Constant learning rate. Policy oscillated badly -- learned something, then unlearned it.
- Failed to converge.

**Run: "v2_tuned" (2M steps, linear LR + clip decay)**
- Reward: -126 -> -88 (best at ~1.35M steps) -> froze at -97
- Learning rate decayed to near-zero too quickly, killing further improvement.
- Better than constant LR but the decay schedule was too aggressive.

**What failed:** Both runs showed the policy was fragile. Gains were temporary. The LR schedule matters a lot -- too high oscillates, too low freezes.

---

### 17:40 -- Phase 3: Self-Play + Removing VecNormalize

Discovered a critical issue: SB3's `VecNormalize` wrapper was causing train/eval mismatch. Normalization statistics drifted during training, so evaluation checkpoints produced different behavior depending on when they were loaded.

**Fix:** Removed `VecNormalize` entirely. Since observations are already pre-normalized to [-1, 1] in the Rust engine, the Python wrapper doesn't need to touch them. This made models fully portable -- no `.pkl` normalization files to track.

**Run: Clean self-play (5M steps)**
- Warmup (vs scripted AI): hit -9.34 (best seen so far)
- Self-play phase: settled around -88

**Key insight:** Pre-normalizing observations in the engine is strictly better than normalizing in Python. Portable models, no drift, no mismatch. This decision paid off for every subsequent run.

---

### 18:08 -- Phase 4: Self-Play 10M Breakthrough

Scaled up to 10M steps with tuned entropy.

**Config:** ent_coef=0.005, constant LR 3e-4.

**Results:**
- Warmup phase: +18 -> +26 -> +38 (steadily climbing)
- Self-play phase: oscillated for a long time, then breakthrough at 10M
- **Final eval reward: +40.37** (std dropped from 1.13 to 0.42)

The agent consistently beat the scripted AI. Evaluation reproduced perfectly without any normalization wrappers -- confirming the Phase 3 decision was correct.

**Best model:** `runs/1v1_selfplay_10M_2026-05-05_18-08-08/best_model/best_model.zip` (obs shape 33, open arena, no obstacles)

This was the first real success. The agent learned to track, aim, and shoot.

---

### 18:42 -- Phase 5: Line-of-Sight, Audio, and Obstacles

Added environmental complexity to make the game tactically interesting.

**New features:**
- **Line-of-sight raycasting:** Enemies behind walls are zeroed out in observations. No more wallhacks.
- **Audio channel:** Shot bearing + shot proximity. Agents can "hear" gunshots through walls even when they can't see the shooter.
- **Obstacles:** Center pillar + side walls added to arena config.
- **Variable strafe AI:** Scripted opponent cycles left/right/push on a 90-tick period.

**Problem: Scripted AI pathfinding with obstacles.**

This took several iterations and all of them failed before the last:

1. **Sweep pattern** -- hack that had the AI sweep in arcs. Didn't work; agents got stuck on geometry.
2. **Potential field** -- attractive force toward enemy, repulsive force from walls. Got permanently stuck on the center pillar (local minimum).
3. **Context steering** -- sampled directions, weighted by goal + obstacle avoidance. Agents walked to opposite sides of the arena and missed each other entirely.
4. **Random waypoint patrol + context steering** -- finally worked. AI patrols random waypoints, uses context steering for obstacle avoidance, switches to pursuit when enemy is visible. Verified: 5 kills in 11 seconds.

**What I learned:** Pathfinding in a physics engine is harder than it looks. Pure reactive approaches (potential fields, steering) fail in enclosed spaces. You need some form of exploration (random waypoints) to break out of degenerate orbits.

---

### 19:00 -- Phase 6: Turn Rate

Added configurable turn rate to make aiming a skill rather than a free action.

**Implementation:**
- Default turn rate: 8 rad/s
- A 180-degree turn takes ~0.4 seconds
- `facing_system` rate-limits rotation toward the desired angle each tick
- Enables flanking as a viable tactic (getting behind someone matters now)

No training run in this phase -- this was a mechanical change integrated into the next run.

---

### 19:24 -- Phase 7: Full Feature Training

First training run with all new features combined: LOS, audio, turn rate, obstacles, context-steering AI.

**Run: "1v1_full_v1" (10M steps)**
- Warmup: peaked at +26
- Self-play: oscillated between -17 and -3, ended around -6
- Best model evaluated cleanly at +26.35

**Best model:** `runs/1v1_full_v1_2026-05-05_19-24-40/best_model/best_model.zip` (obs shape 35)

**Critical problem discovered:** The model was trained exclusively as agent 0 (team 0, always spawning at the same position). When placed on agent 1, performance was garbage. The policy had memorized its spawn position and team ID rather than learning generalizable combat behavior.

---

### 20:00 -- Phase 8: Symmetry Fixes

Addressed the team bias problem from Phase 7.

**Attempted fixes (in order):**
1. Removed team from observations entirely -- broke 2v2 compatibility (team info is needed for multi-agent scenarios).
2. Added random `agent_id` assignment: learner randomly plays as blue or red each episode, in both the gym wrapper and the self-play wrapper.
3. Restored team to observations (needed for future 2v2 support).

**Additional randomization:**
- Spawn point pool: 8 positions scattered around the arena (instead of fixed symmetric spawns)
- Obstacle jitter: +/-40px positional noise per episode
- Obstacle rotation: random 90-degree swaps each episode
- Added `RoundStart` telemetry event broadcasting obstacle positions + spawn points to the web viewer

The goal was to make it impossible for the policy to memorize any single layout.

---

### 20:19 -- Phase 9: Full Randomization Training

Trained with all randomization active.

**Run: "1v1_final_10M" (10M steps)**
- Warmup: -31 -> -12
- Self-play: -37 -> -8 (best) -> finished at -31
- Eval variance: +/-7 to +/-17 (expected with randomized layouts)
- **Never broke positive.**

**What happened in the viewer:** Agents got stuck in corners, pushed each other around without shooting, showed no clear combat strategy.

The model learned fragments of combat behavior but couldn't consistently put them together in randomized environments. The problem space had grown too large too fast.

---

### 20:45 -- Phase 10: Research and Diagnosis

Stepped back to study what works in FPS RL literature (DeepMind FTW, OpenAI Five methodology, VizDoom competition winners).

**Identified three critical design flaws:**

1. **Absolute angle action space.** The continuous `look_angle` action outputs a target angle in [-PI, PI]. This forces the network to learn coordinate geometry before it can aim -- it needs to compute "enemy is at bearing X, I need to output angle X" as an implicit trig function. This is a known convergence killer in the RL-for-games literature. **Fix:** Replace with discrete relative turn bins (e.g., turn left 15 degrees, turn right 15 degrees, no turn). The network only needs to learn "turn toward the thing I want to shoot."

2. **No existence penalty.** The reward function gives 0 for surviving and -1 for dying. This makes doing nothing (camping) weakly optimal -- you can't die if you never engage. **Fix:** Add a small per-tick penalty (e.g., -0.001) so that passive play accumulates cost and only offensive play can break even.

3. **Naive self-play.** Both the learner and opponent always use the latest policy. This creates a degenerate feedback loop: both agents converge to the same strategy, then co-adapt into mutual incompetence (e.g., both learn to camp because neither attacks). **Fix:** Maintain an opponent checkpoint buffer -- sample opponents from past policy snapshots to force the learner to generalize against diverse play styles.

---

### Phase 11: Research Fixes Applied (~21:00)

Three research-backed fixes implemented:
1. **Discrete relative turn bins** (7 bins: hard-left to hard-right) replacing continuous absolute angle — agent learns "turn left a bit" not "set angle to 1.23 radians"
2. **Existence penalty** (-0.0005/tick) — camping is always-negative reward, forces engagement
3. **Opponent checkpoint buffer** (pool of 20 past selves, randomly sampled) — prevents both agents converging to same degenerate strategy

Results: Best run yet.
- Positive from first eval (+10 at 500k)
- Warmup peaked at +17, self-play peaked at +22.68
- **Never went negative** — first run to stay positive through entire 10M self-play
- Final model at +1.2 after 10M steps (break-even against scripted AI after fighting 20 versions of itself)
- Confirmed reproducible across two runs with near-identical curves

---

### Phase 12: Live Training Viewer (~21:00)

Built `--live-view` flag for train_selfplay.py:
- Separate spectator thread runs its own env at real-time speed
- Loads model snapshots every N steps (configurable via `--live-snapshot-freq`)
- Streams telemetry to WebSocket on port 3000
- React viewer connects normally and shows live training matches
- Zero impact on training throughput — completely decoupled

---

### Phase 13: Navigation Gap Identified (~21:30)

Despite +22 eval scores, the model can't navigate around obstacles. Investigation revealed:
- The +22 came from lucky spawns where both agents had clear line-of-sight
- During warmup, the scripted AI always hunted the RL agent (context steering). The RL agent never needed to learn search behavior.
- During self-play, neither agent knows how to search → both circle aimlessly behind obstacles
- The agent learned combat but not navigation

**Root cause**: Threw too much complexity in at once. The agent can't learn combat AND navigation simultaneously from scratch.

**Fix**: Curriculum learning (should have done this from the start):
- Phase 1: No obstacles, fixed spawns facing each other → pure combat training
- Phase 2: Resume from phase 1, add obstacles → agent already knows how to fight, just learns navigation
- Phase 3: Add randomization (spawn positions, obstacle layouts)

---

### Key Learning

Curriculum > complexity dumping. The research was clear about this but we ignored it. Adding obstacles, random spawns, random obstacles, spawn pools, and LOS all at once meant the agent couldn't learn ANY skill well. Start with the simplest version of the problem, prove the agent can solve it, then add one difficulty dimension at a time.

---

## Key Learnings

- **Pre-normalize observations in the engine**, not in Python. Portable models, no normalization state drift, clean eval.
- **Start simple, add complexity gradually.** Phase 4 (open arena, no obstacles) succeeded. Phase 9 (everything at once) failed. Curriculum matters.
- **Absolute angle actions are terrible for FPS.** Use relative discrete turn bins. The network shouldn't need to be a trigonometry solver.
- **VecNormalize causes more problems than it solves** when observations are already pre-normalized.
- **Self-play needs opponent diversity.** Latest-self vs latest-self converges to degenerate equilibria. Use a checkpoint buffer.
- **An existence penalty is essential** to prevent passive camping strategies.
- **Turn rate makes the game more interesting** but also harder to learn -- it's a worthwhile tradeoff for realism.
- **Pathfinding for scripted AI in enclosed spaces** needs exploration (random waypoints), not just reactive steering.

---

## Best Models Summary

| Model | Eval Reward | Obs Shape | Features |
|-------|-------------|-----------|----------|
| `runs/1v1_selfplay_10M_2026-05-05_18-08-08/best_model/best_model.zip` | +40.37 | 33 | Open arena, no obstacles, old obs format |
| `runs/1v1_full_v1_2026-05-05_19-24-40/best_model/best_model.zip` | +26.35 | 35 | Obstacles, LOS, audio, turn rate |

---

## Next Steps (Identified, Not Yet Implemented)

- Replace absolute look_angle with discrete relative turn bins
- Add per-tick existence penalty to reward function
- Implement opponent checkpoint buffer for self-play
- Consider curriculum training: master open arena first, then introduce obstacles
- Investigate frame stacking or recurrent policies for partial observability
