# Game Bot RL System — Full Build Spec

A complete architecture and training plan for a top-down 2D shooter bot using hierarchical RL with teacher-student transfer.

---

## System Overview

The system has five distinct components built and trained in sequence. Each one is a separate concern with its own inputs, outputs, and training process. You don't build them all at once — each phase produces a working system before the next begins.

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING ENVIRONMENT                  │
│                  (Your Rapier 2D Sim)                    │
│                                                         │
│  Physics · Raycasts · LOS · Spawning · Weapons · Maps   │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
        Perfect State               Rendered Frames
        (god mode)                  (pixel output)
               │                          │
               ▼                          ▼
     ┌─────────────────┐        ┌─────────────────┐
     │  TEACHER AGENT   │        │  STUDENT AGENT   │
     │  (privileged)    │───────▶│  (vision-based)  │
     │                  │ distil │                  │
     └─────────────────┘        └─────────────────┘
               │
     Composed of:
               │
     ┌─────────┴──────────────────────────────────┐
     │                                             │
     ▼                  ▼                 ▼        ▼
┌──────────┐   ┌──────────────┐   ┌────────┐  ┌────────┐
│Navigation│   │   Tactical   │   │Weapon  │  │Strategy│
│  (A*)    │   │    Model     │   │Selector│  │ Model  │
│ classical│   │    (RL)      │   │(part of│  │  (RL)  │
│ not      │   │              │   │tactical│  │optional│
│ learned  │   │              │   │model)  │  │        │
└──────────┘   └──────────────┘   └────────┘  └────────┘
```

---

## Component 1: The Simulation Environment

This is your Rapier 2D engine. It needs to provide the following capabilities before any training begins.

### Required Sim Features

**Physics and movement:** Character bodies with velocity, collision, friction. Obstacle geometry (walls, crates, cover objects). Projectile physics (bullets as raycasts or fast-moving bodies). Grenade physics with bounce, travel time, and area-of-effect damage.

**Information extraction (god mode state):** Every tick, the sim must be able to export a complete state snapshot. This is the training data for the teacher. The state includes:

- Bot position (x, y), velocity (vx, vy), facing angle, health, shield (if applicable)
- Per weapon: current ammo count, reload state (is reloading, ticks remaining), cooldown ticks remaining, weapon type enum
- Per enemy (up to N enemies): absolute position, velocity, health, facing angle, current weapon, is-visible boolean, time-since-last-seen (in ticks), last-known-position if not visible
- Raycasts: 64 rays cast from bot position in evenly-spaced directions (every 5.625 degrees for full 360). Each ray returns: distance to first hit (normalised 0-1 where 1 is max ray length), entity type of hit (0=nothing, 1=wall, 2=enemy, 3=projectile, 4=pickup)
- Per-enemy line-of-sight: boolean, computed by casting a ray from bot to enemy and checking if it hits geometry first
- Cover analysis for candidate positions: for each of 12 candidate move targets (see action space below), pre-compute: does this position block LOS from nearest enemy? Distance to nearest wall. Number of enemy LOS rays blocked

**Information extraction (pixel mode):** The sim should be able to render a top-down frame at a fixed resolution (64x64 or 84x84 pixels is standard for RL vision). This is used later for training the student. The render should show: obstacles, bot, enemies, projectiles, pickups. Minimal visual detail is fine — solid colors, no textures needed. You'll add visual randomization later.

**Environment reset and parallelism:** The sim must support fast reset (new episode in <1ms). Ideally support running N environments in parallel (vectorized envs). 64-256 parallel envs is typical for PPO training. Since you're running headless Rapier, this should be straightforward — just instantiate N world instances.

**Map variety:** Start with 3-5 hand-designed maps of increasing complexity. Map 1: open arena, no obstacles. Map 2: a few scattered boxes. Map 3: corridors and rooms. Map 4: asymmetric cover positions. Map 5: multi-room with chokepoints. Training should randomly sample maps per episode.

---

## Component 2: Navigation (A* Pathfinding)

This is classical, not learned. Already built. But a few things to get right for integration with the RL agent.

### Navigation Grid

Convert your map geometry into a 2D grid (cell size ~0.5m or whatever your character width is). Mark cells as walkable or blocked. Run A* on this grid when the tactical model requests a move-to target.

### Candidate Position Generation

Every tick, generate 12 candidate positions the tactical model can choose from:

- 8 compass directions (N, NE, E, SE, S, SW, W, NW) at a fixed distance from current position (e.g., 3 meters). If a compass position lands inside a wall, snap it to the nearest walkable cell in that direction
- "Stay" (current position)
- "Nearest cover from enemy 0" — the closest position that blocks LOS from the primary threat. Pre-compute this by testing cells near the bot
- "Advance toward enemy 0" — position that moves toward the primary threat while maintaining nearest wall adjacency (aggressive flanking position)
- "Retreat to spawn / safe zone" — a pre-defined fallback position per map

For each candidate, pre-compute and attach these features:
- Distance from current position (A* path length, not euclidean)
- Does this position have LOS to enemy 0? (boolean)
- Is this position in cover from enemy 0? (boolean — is there a wall between position and enemy?)
- Distance from this position to enemy 0
- Number of enemies with LOS to this position

These features get fed to the tactical model as part of the state. The model doesn't need to imagine what a position is like — you tell it.

### Path Execution

When the model picks a candidate, A* generates a path. The bot follows the path one step per tick. Key behaviors:

- If the model picks a new target before the current path completes, abandon the old path and re-route
- The bot should still be able to aim and shoot while walking a path — movement and combat are simultaneous
- If the path is blocked by a dynamic obstacle (another player), re-route automatically
- Path following should smooth corners slightly to look natural (optional, cosmetic)

---

## Component 3: The Tactical Model (Teacher Version)

This is the core RL agent. It runs every tick and makes all combat and movement decisions.

### Network Architecture

A standard MLP (multi-layer perceptron) is fine for structured state input. You don't need a CNN for the teacher — it sees numbers, not pixels.

```
Input: state vector (see below)
  → Dense layer (256 units, ReLU)
  → Dense layer (256 units, ReLU)
  → Dense layer (128 units, ReLU)
  → Split into output heads:
      Head 1: move_target (12 options, softmax → discrete choice)
      Head 2: aim_angle (continuous, tanh → scaled to -π to π)
      Head 3: shoot (binary, sigmoid → 0 or 1)
      Head 4: weapon_select (N weapons, softmax → discrete choice)
```

Shared trunk with multiple output heads. The heads share the hidden layers so the model can learn correlations (e.g., "I selected grenade AND I'm aiming at the wall" go together).

### State Vector (Teacher Input)

Concatenate all of these into a single flat vector:

**Self state (fixed size):**
- Position x, y (normalised to map bounds, 0-1)
- Velocity vx, vy (normalised)
- Health (0-1)
- Facing angle (sin, cos encoding — two values, avoids discontinuity at ±π)
- Per weapon (×N weapons): ammo fraction (0-1), is-reloading (0/1), cooldown fraction (0-1)
- Current weapon (one-hot, size N)

**Enemy state (per enemy, fixed max enemies, zero-pad if fewer):**
- Relative position dx, dy (normalised)
- Relative velocity dvx, dvy
- Health (0-1)
- LOS boolean (0/1)
- Time since last seen (normalised, 0-1 capped at some max like 300 ticks)
- Last known position dx, dy (relative, if not currently visible)
- Threat score: a simple heuristic like (1/distance) × LOS × enemy_health. Pre-computed, saves the model effort

**Raycast data (64 rays × 2 values = 128 values):**
- Distance (0-1)
- Hit type (could one-hot encode, or just use the integer 0-4)

**Candidate position features (12 positions × 5 features = 60 values):**
- Path distance (normalised)
- Has LOS to primary enemy (0/1)
- Is in cover from primary enemy (0/1)
- Distance to primary enemy from that position (normalised)
- Number of enemies with LOS (0-1 normalised by max enemies)

**Context (small):**
- Round timer (0-1, fraction of round elapsed)
- Score differential (normalised)
- Strategic goal (one-hot, only used when strategic model is active — otherwise zeros)

Total state size: roughly 250-350 values depending on max enemies and weapons. This is tiny for a neural network.

### Action Space Summary

| Output | Type | Range | What it does |
|--------|------|-------|-------------|
| move_target | Discrete (12) | 0-11 | Which candidate position to move toward |
| aim_angle | Continuous | -π to π | Direction to point/aim |
| shoot | Binary | 0 or 1 | Fire current weapon this tick |
| weapon_select | Discrete (N) | 0 to N-1 | Which weapon to equip |

### Training Algorithm

Use PPO (Proximal Policy Optimization). It's the standard for this kind of setup — stable, works with mixed discrete/continuous actions, parallelizes well.

**Hyperparameters (starting points, tune from here):**
- Learning rate: 3e-4, linearly decayed to 0 over training
- Clip ratio: 0.2
- Entropy coefficient: 0.01 (encourages exploration, decay to 0.001 over time)
- GAE lambda: 0.95
- Discount gamma: 0.99
- Minibatch size: 4096
- Epochs per update: 10
- Parallel environments: 64-256

**Framework options:** Stable Baselines 3 (Python, easiest to get started), CleanRL (Python, minimal and readable), or roll your own in Rust if you want speed and already have the sim in Rust.

---

## Component 4: Training Curriculum

Don't train everything at once. Phase it.

### Phase 1 — Aim and Shoot (open arena, no obstacles)

**Map:** Open arena, no cover, no walls except boundaries.

**Enemies:** 1 scripted bot that moves randomly and shoots back.

**Action space:** Aim + shoot only. Movement is random or fixed. Weapon is fixed (rifle only).

**Rewards:**
- +1.0 for killing the enemy
- -1.0 for dying
- +0.1 for dealing damage (per hit)
- -0.1 for taking damage (per hit)
- +0.01 per tick for aiming within 10 degrees of enemy (aim shaping)
- -0.0005 per tick (time pressure — end the fight)

**Duration:** Train until the bot reliably beats the scripted bot >80% of rounds. This is where you already are.

**Expected training time:** ~1-5M timesteps depending on sim speed.

### Phase 2 — Movement and Positioning (obstacles, no weapons)

**Map:** Maps 2-4 with obstacles. Randomly selected per episode.

**Enemies:** 1 scripted bot that shoots but doesn't move much (stands in a fixed position or patrols).

**Action space:** Move target + aim + shoot. Still single weapon.

**Rewards (keep Phase 1 rewards, add):**
- +0.3 for dealing damage while in cover (incentivizes positioning before shooting)
- +0.05 for each new grid cell visited in the last 100 ticks (anti-camping / exploration bonus)
- -0.02 per tick if stationary and an enemy is alive and not in LOS (you're hiding uselessly)
- +0.1 for acquiring LOS to an enemy you didn't have LOS to 10 ticks ago (reward for maneuvering to get a shot)

**Key moment:** This is where the model learns that moving to a better position before shooting is worth more than shooting from a bad position. The "damage from cover" bonus is the critical reward.

**Expected training time:** ~5-20M timesteps. This phase is harder because the exploration space is larger.

**How to know it's working:** Watch replays. The bot should start circling around obstacles to get angles on the enemy rather than running straight at them or hiding behind cover indefinitely.

### Phase 3 — Weapon Selection

**Map:** All maps, randomly selected.

**Enemies:** 1-2 bots. Mix of scripted and frozen copies of the Phase 2 model (self-play begins here).

**Action space:** Full — move + aim + shoot + weapon select.

**Rewards (keep all previous, add):**
- No weapon-specific rewards needed. The existing kill/damage rewards are sufficient
- The model will discover grenade bouncing, shotgun close-range, rifle long-range naturally
- Optionally: +0.05 for switching weapons (mild incentive to not just use rifle forever during early exploration)

**Important:** Load the Phase 2 weights as the starting point. Don't train from scratch. The model already knows how to move and aim — now it just needs to learn weapon dynamics on top.

**Expected training time:** ~5-10M additional timesteps.

### Phase 4 — Self-Play and Generalization

**Map:** All maps.

**Enemies:** Copies of the current model at various checkpoints (league training). Keep a pool of past model snapshots. Each episode, the opponent is randomly drawn from: latest model (50%), random past snapshot (30%), scripted bot (20%).

**Rewards:** Same as Phase 3. No changes.

**Purpose:** This prevents the model from overfitting to one opponent's playstyle. League training forces robust behavior. The scripted bot baseline prevents catastrophic forgetting of fundamentals.

**Expected training time:** ~20-50M timesteps. This is the long grind but produces the strongest agent.

---

## Component 5: Strategic Model (Optional)

Only build this if you're doing team modes or objective-based games. Skip for 1v1 deathmatch.

### What it is

A separate, smaller neural network that runs on a slower clock (every 30-60 ticks, roughly 0.5-1 seconds of game time).

### Network Architecture

```
Input: macro state vector
  → Dense layer (128 units, ReLU)
  → Dense layer (64 units, ReLU)
  → Output: goal (discrete, 1 of 6-8 options, softmax)
```

### Macro State Vector

- Per-team health totals (0-1)
- Per-team alive counts
- Objective status (flag position, capture progress, etc.)
- Map control heuristic (what fraction of the map does each team "own" based on recent positions)
- Round timer
- Score differential
- Average team position (centroid)
- Average enemy position (centroid, if any are visible)

### Goal Output Options

| Goal ID | Meaning | Effect on Tactical Model |
|---------|---------|------------------------|
| 0 | Fight | No position bias, pure combat |
| 1 | Push objective | Candidate positions biased toward objective |
| 2 | Hold position | Candidate "stay" option gets bonus weight |
| 3 | Flank left | Candidate positions on left side of enemy get bonus |
| 4 | Flank right | Same, right side |
| 5 | Retreat | Retreat candidate gets strong bonus |
| 6 | Regroup | Move toward nearest teammate |
| 7 | Support | Move toward teammate with lowest health |

The goal is fed to the tactical model as a one-hot vector appended to its state input. The tactical model was already trained with this input slot (filled with zeros during Phases 1-4). Now it has real values.

### Training the Strategic Model

**Freeze the tactical model completely.** The strategic model is trained with RL (PPO again), but on a much longer timescale.

**Episode structure:** Full rounds, not individual fights. The strategic model gets rewarded on round-level outcomes.

**Rewards:**
- +1.0 for winning the round
- -1.0 for losing the round
- +0.1 per objective captured
- +0.01 per tick where team has map control advantage
- No rewards for individual kills (that's the tactical model's problem)

**Expected training time:** ~10-30M timesteps of game time (which is fewer strategic model updates since it only acts every 30-60 ticks).

---

## Component 6: Teacher-Student Distillation

This is how you transfer the god-mode teacher to a vision-only student for deployment to a real game.

### Step 1: Generate Training Data

Run the fully-trained teacher (Phases 1-4 complete) in your sim. For every tick, record:

**The teacher's state input** (the full god-mode vector — you don't use this for the student, it's just for reference/debugging).

**The teacher's actions** (move target, aim, shoot, weapon — this is the training target).

**A rendered frame** from the sim (64x64 or 84x84 pixels, top-down view centered on the bot). This is what the student will see.

**Basic proprioceptive state** that would be available in a real game without reverse engineering: own health, own ammo, own velocity, own position (from HUD or minimap). Anything the real game's UI shows the player.

Collect ~1-5M frames across varied maps, opponents, and situations. Bias toward interesting situations (fights, flanking, grenade usage) rather than boring ones (walking across empty map). You can do this by only recording ticks where an enemy is within sensor range.

### Step 2: Train the Vision Encoder

The student network has a different architecture because it processes pixels:

```
Input: rendered frame (84x84x3 RGB)
  → Conv2d(32 filters, 8x8, stride 4) → ReLU
  → Conv2d(64 filters, 4x4, stride 2) → ReLU
  → Conv2d(64 filters, 3x3, stride 1) → ReLU
  → Flatten → Dense(256)
  → Concatenate with proprioceptive state vector
  → Dense(256, ReLU)
  → Dense(128, ReLU)
  → Same output heads as teacher (move, aim, shoot, weapon)
```

This is a standard Nature DQN-style CNN backbone feeding into the same decision heads.

### Step 3: Train via Imitation

**Loss function:** For each recorded frame, the student predicts actions. Compare to teacher's actions:

- Move target: cross-entropy loss (discrete classification)
- Aim angle: MSE loss or cosine similarity loss
- Shoot: binary cross-entropy
- Weapon: cross-entropy

Combined loss: weighted sum of the four. Weight aim and move higher since they matter most.

**Training:** Standard supervised learning. Adam optimizer, lr=1e-4, batch size 256, train for ~50-100 epochs over the dataset. Validate on a held-out set of frames.

### Step 4: Fine-Tune with DAgger

Pure imitation learning has a compounding error problem — the student makes a small mistake, ends up in a state the teacher never demonstrated, and makes bigger mistakes. DAgger fixes this.

Run the student in the sim. At each tick, also query the teacher (which has access to the perfect state) for what it would do. Record these as new training examples. Add them to the dataset. Retrain the student. Repeat 5-10 rounds.

This is the most important step for closing the gap between teacher and student performance.

### Step 5: Domain Randomization (for real game transfer)

When generating frames from your sim for student training, randomize the visual appearance:

- Randomly change obstacle colors each episode
- Randomly change character sprites/colors
- Add random background textures
- Vary lighting/brightness
- Add noise to the rendered image
- Randomly scale objects ±20%
- Occasionally occlude parts of the frame (simulates HUD elements in a real game)

This forces the vision encoder to learn spatial features (shapes, positions, movement) rather than memorizing specific pixel patterns. When you later show it frames from a real game that looks completely different, the encoder can still extract the relevant spatial information.

### Step 6: Real Game Fine-Tuning

Capture frames from the actual target game. Manually label a small set (~1000-5000 frames) with the structured state: enemy positions, obstacle positions, LOS. Fine-tune only the CNN layers of the student on this labeled data while keeping the decision heads frozen.

Alternatively, if you can extract game state through memory reading or API, skip the vision approach entirely and just map the game's state format to your model's expected input format. This is faster, more reliable, and what most practical game bot projects actually do.

---

## Training Infrastructure Notes

### Hardware Requirements

The teacher training (Phases 1-4) is CPU-bound if your sim is CPU-based (Rapier is). The neural network is tiny — the bottleneck is running 256 parallel sim instances. A modern 8-core CPU handles this fine. GPU helps for the network updates but isn't strictly necessary for an MLP this small.

The student training (vision) is GPU-bound because of the CNN. A single decent GPU (RTX 3070+) is plenty. You have GODZILLA so this is fine.

### What to Log

Log everything. Specifically:

- Mean episode reward per training phase
- Win rate against scripted bot (should only go up)
- Average episode length (should decrease as bot gets more efficient at killing)
- Kill/death ratio
- Distance traveled per episode (should increase in Phase 2 when movement is added)
- Weapon usage distribution (in Phase 3, should diversify from 100% rifle)
- Heatmaps of bot positions on each map (should show intelligent positioning, not clustering)

### When to Stop Each Phase

**Phase 1:** Win rate >80% against scripted bot AND reward curve has plateaued for >500k timesteps.

**Phase 2:** Win rate >70% on maps with obstacles AND the bot visibly navigates around cover (watch replays, this is a qualitative check).

**Phase 3:** The bot uses at least 2 weapons regularly AND win rate hasn't dropped from Phase 2 levels.

**Phase 4:** Self-play Elo has stabilized AND the bot still beats the scripted bot >90% (sanity check for catastrophic forgetting).

---

## The Whole Pipeline as a Checklist

1. Build sim with full state export, raycasts, LOS computation, and candidate position pre-computation
2. Implement A* navigation with the 12-candidate system and path execution
3. Define state vector and action space exactly as specified
4. Train Phase 1: aim + shoot in open arena vs scripted bot
5. Verify Phase 1 works. Watch replays. Check win rate
6. Train Phase 2: add movement on obstacle maps. Load Phase 1 weights
7. Verify Phase 2 works. Watch replays. Bot should use cover
8. Train Phase 3: add weapon selection. Load Phase 2 weights
9. Verify Phase 3 works. Check weapon usage distribution
10. Train Phase 4: self-play league. Long grind. Monitor Elo
11. (Optional) Train strategic model for team/objective modes
12. Add pixel rendering to sim with domain randomization
13. Generate imitation dataset from teacher
14. Train student CNN via supervised imitation
15. Fine-tune student with DAgger (5-10 rounds)
16. Deploy student to target environment (vision or API depending on target)

Each numbered step is a discrete milestone. Don't skip ahead. Each step should produce measurably better behavior than the last.