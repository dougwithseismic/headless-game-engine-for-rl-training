# Pokemon Gold RL Training Results

## Current Best: Milestone Reward (Iteration 3)

The agent navigates west from New Bark Town to Route 29. This is verified via honest stochastic eval, not training rollout metrics.

### Setup

| Parameter | Value |
|-----------|-------|
| ROM | Pokemon Gold (USA, SGB Enhanced) |
| Save state | New Bark Town, Lv 5 Cyndaquil, 20/20 HP |
| Observation | 25 RAM features + 8 temporal deltas = 33-dim (or + 1440 screen pixels with `--screen`) |
| Action space | MultiBinary(6) -- up/down/left/right/A/B (Start/Select removed) |
| Reward | MilestoneReward (map milestones + count-based exploration + dialogue + level/badge) |
| Wrappers | TemporalObsWrapper + AntiLoopWrapper (softened) + PokemonTelemetryWrapper |
| Policy | MlpPolicy (256x256) or MultiInputPolicy (CNN + MLP) with `--screen` |
| Gamma | 0.998 |
| Hardware | M4 Max 64GB, CPU only |

### Honest Eval Results (20K steps, milestone reward)

Evaluated with stochastic policy (`deterministic=False`), 5 episodes per checkpoint.

| Metric | Range | Notes |
|--------|-------|-------|
| Unique tiles | 150-208 | Per episode |
| Maps visited | 3.4-4.0 | New Bark Town + Route 29 + interiors |
| Movement rate | 47-93% | Percentage of steps where position changed |
| Route 29 reached | Yes (all checkpoints) | Verified via map_number == 3 in eval |
| Wild battles | 0-2 | Accidental, from tall grass on Route 29 |
| Throughput | ~300 steps/sec (CNN+RAM), ~700 steps/sec (RAM-only) | 1 env, M4 Max |

The agent consistently walks west through New Bark Town, exits to Route 29, and explores. It does not navigate strategically -- it follows the milestone reward gradient toward maps it hasn't visited.

### Milestone Reward Breakdown

| Signal | Reward | Trigger |
|--------|--------|---------|
| Reach Route 29 | +50.0 | First time map (24,3) is entered (once per episode) |
| Tile exploration | 1.0/sqrt(N) | Each tile visit, diminishing returns |
| Level up | +3.0 per level | High-water-mark tracked |
| Badge earned | +20.0 per badge | High-water-mark tracked |
| New party member | +5.0 | High-water-mark tracked |
| Dialogue advance | +0.1 | Press A when textbox active |

### Anti-Loop Configuration (Softened)

As configured in `train_gold.py` (5x softer than default wrapper values):

| Pattern | Threshold | Penalty |
|---------|-----------|---------|
| Button spam | 5+ same action | -0.02 |
| Heavy spam | 10+ same action | -0.05 |
| Action cycling | Window of 8 | -0.03 |
| Idle (no buttons) | 15+ steps | -0.05 |

These are mild enough that exploration reward drowns them out. They nudge the agent away from degenerate patterns without dominating the reward signal.

## Training History

### Iteration 1: Exploration Reward, 8 Buttons, 2M Steps

The initial run used ExplorationReward with 8-button action space (including Start/Select).

**What the metrics showed**: 449 tiles, 6 maps, 20 wild battles at peak.

**What was actually happening**: These were training rollout metrics (stochastic policy with entropy). The deterministic eval policy just walked into a wall. Training metrics showed entropy-driven randomness, not learned behavior. Additionally, 95% of steps were trapped in menus from random Start/Select presses.

**Result**: Misleading. The agent didn't learn to explore -- random button presses with entropy produced exploration-looking metrics during training.

### Iteration 2: 6 Buttons, CNN+RAM, Honest Eval

Fixed the two critical problems:

1. **Removed Start/Select** (6-button action space). Menu trap rate dropped from 95% to 0%.
2. **Added stochastic eval** (`deterministic=False`). Shows the actual policy distribution, not the collapsed argmax.

| Config | Unique positions | Maps | Movement % |
|--------|-----------------|------|------------|
| 8 buttons, random | 1 | 1 | 5% (stuck in menus) |
| 6 buttons, random | 41 | 2 | 34% |
| 6 buttons, 100K CNN+RAM | 113 | 3-4 | 41% |

Better, but still no clear directional behavior. The agent explored nearby tiles but didn't head toward Route 29 specifically.

### Iteration 3: Milestone Reward (Current)

Added MilestoneReward with a +50 bonus for reaching Route 29. The agent now has a directional signal -- it gets a large reward for going west into Route 29.

At 20K steps the agent reaches Route 29 in all eval checkpoints. This is 100x fewer steps than iteration 1 (which never reliably reached Route 29 at all).

The milestone reward works because it combines:
- A large one-time bonus for reaching the target map (directional signal)
- Diminishing-return exploration for intermediate tiles (keeps moving)
- Dialogue bonus (clears NPC text that blocks movement)
- No negative penalties for "wrong" behavior (positive-only)

## Bugs Found and Fixed

### 1. GBC WRAM Bank Reads

**Bug**: `pyboy.memory[0xDA03]` reads the currently-mapped WRAM bank, which may not be bank 1 during GBC CPU operations. Gold stores all game state in WRAM bank 1.

**Fix**: Use explicit banked reads `pyboy.memory[1, addr]` for all addresses in 0xD000-0xDFFF range. Applied in `PyBoyObservationSource.read()` and `PyBoyHost.read_memory()`.

### 2. PyBoy Speed Limiter

**Bug**: `tick(count=24)` batches 24 frames but PyBoy only applies the speed limiter once per call, not per frame. Headed mode ran at ~12x speed instead of 1x.

**Fix**: `PyBoyHost.tick()` loops frame-by-frame when speed limiting is active (`speed > 0` and `render=True`). Headless path unchanged (still uses fast batch).

### 3. False Telemetry Events (RAM Glitches)

**Bug**: GBC WRAM bank switching during CPU operations causes momentary zero-reads. A value like `party1_level` would briefly read 0 then recover to 5. The frame-to-frame delta detection saw 0->5 as a "level up", producing 100-190 false level_ups per episode.

**Fix**: High-water-mark tracking. Level/badge/party events only fire when the new value exceeds the maximum seen so far. A glitch (5->0->5) never exceeds the max of 5, so no false event. Applied in both `PokemonTelemetryWrapper` and `MilestoneReward`.

### 4. Reward Hacking -- Agent Learned to Stand Still

**Bug**: ExplorationReward gave +0.3 for new tiles and -0.01 for revisits. With ~50 reachable new tiles and ~2000 revisits per episode, the math was: +15 from new tiles, -20 from revisits = net negative. The rational strategy became "don't move at all."

**Fix**: Removed all revisit penalties. Switched to count-based exploration (1/sqrt(N)) that never goes to zero. Increased new tile bonus to +2.0. Softened anti-loop penalties (5x less severe than defaults). Removed HP and battle penalties.

### 5. 2-Tile Oscillation Exploit

**Bug**: Agent discovered it could oscillate between two tiles (A->B->A->B) for constant reward.

**Fix**: Anti-bounce detection in ExplorationReward -- returning to the previous tile position gives zero reward. Combined with cycle detection in AntiLoopWrapper.

### 6. Save State Corruption

**Bug**: A cleanup script overwrote the save state with one stuck in dialogue. Agent couldn't move at all.

**Fix**: Backup system added (`data/pokemon_gold_ready_backup.state`). `save_gold_state.py` script for creating clean states.

### 7. 95% Menu Trap with 8-Button Action Space

**Bug**: Random Start/Select presses constantly opened menus. The agent read (0,0) coordinates because the game was in menu state. 95% of steps were effectively blind.

**Fix**: Removed Start and Select from the action space. 6 buttons only: up/down/left/right/A/B. Menu transition rate dropped to 0%.

### 8. Dialogue Blocking Movement on Reset

**Bug**: Some save states loaded with active dialogue/textbox. The agent couldn't move until the text was dismissed, wasting the first ~50 steps.

**Fix**: A-press burst after save state load to clear any active dialogue before the episode begins.

### 9. Misleading Training Rollout Metrics

**Bug**: Training rollout metrics (stochastic policy with high entropy) showed impressive exploration numbers (449 tiles, 6 maps) that were just entropy-driven randomness. These were reported as learned behavior.

**Fix**: Added `PokemonEvalCallback` that runs honest stochastic eval (`deterministic=False`) separately from training, logging real game progress (tiles, maps, battles, Route 29 reached). Training metrics and eval metrics are now clearly separated.

## Performance

Tested on M4 Max 64GB:

| Config | Steps/sec |
|--------|-----------|
| 1 env, RAM-only, headless | ~700 |
| 1 env, CNN+RAM, headless | ~300 |
| 1 env, headed (speed=3) | ~2.5 |
| 4 envs, SubprocVecEnv | ~2,500 (estimated) |

Each step advances 24 emulator frames (~0.4s game time). At 700 steps/sec, that's 16,800 emulated frames/sec.

## Reproducing

```bash
cd python

# Train with milestone reward (recommended, ~1 min for 20K steps)
python train_gold.py \
  --state data/pokemon_gold_ready.state \
  --reward milestone \
  --temporal --anti-loop \
  --gamma 0.998 --steps 20000 --envs 1

# Train with CNN+RAM (slower, richer observations)
python train_gold.py \
  --state data/pokemon_gold_ready.state \
  --reward milestone --screen \
  --temporal --anti-loop \
  --gamma 0.998 --steps 100000 --envs 1

# Longer training run
python train_gold.py \
  --state data/pokemon_gold_ready.state \
  --reward milestone \
  --temporal --anti-loop --frame-stack 4 \
  --gamma 0.998 --steps 500000 --envs 4

# Monitor with TensorBoard
tensorboard --logdir runs/gold_milestone_*/tb

# Evaluate a trained model (stochastic, honest)
python train_gold.py \
  --eval runs/gold_milestone_*/best/best_model.zip \
  --state data/pokemon_gold_ready.state \
  --reward milestone \
  --temporal --anti-loop \
  --max-steps 4096 --episodes 5

# Watch the agent play (headed, real-time)
python train_gold.py \
  --eval runs/gold_milestone_*/best/best_model.zip \
  --state data/pokemon_gold_ready.state \
  --reward milestone \
  --temporal --anti-loop \
  --headed --episodes 3
```
