# Pokemon Gold RL Training

Train reinforcement learning agents to play Pokemon Gold (GBC) using the GhostLobby game bridge framework. RAM-based observations (with optional screen pixels), save state resets, milestone-based reward, and full PPO training pipeline.

The agent learns to navigate New Bark Town and reach Route 29. It does not play Pokemon strategically.

## Quick Start

```bash
cd python

# 1. Create a save state (play past the title, close window to save)
python save_gold_state.py

# 2. Train with milestone reward (reaches Route 29 within 20K steps)
python train_gold.py \
  --state data/pokemon_gold_ready.state \
  --reward milestone \
  --temporal --anti-loop \
  --gamma 0.998 --steps 100000 --envs 1

# 3. Watch the trained agent play
python train_gold.py \
  --eval runs/gold_milestone_*/best/best_model.zip \
  --state data/pokemon_gold_ready.state \
  --reward milestone \
  --temporal --anti-loop \
  --headed --episodes 3

# 4. Train with screen observations (slower, CNN+RAM)
python train_gold.py \
  --state data/pokemon_gold_ready.state \
  --reward milestone --screen \
  --temporal --anti-loop \
  --gamma 0.998 --steps 100000
```

## Architecture

```
train_gold.py
  |
  +-- make_pokemon_gold_bridge()     # Profile: ROM + RAM addresses + save state
  |     +-- PyBoyHost                # Shared emulator instance (cgb=True)
  |     +-- PyBoyActionSink          # MultiBinary(6) -> button presses
  |     +-- PyBoyObservationSource   # 25 RAM features via banked WRAM reads
  |     +-- PyBoyReset               # Instant save state resets (~5ms)
  |
  +-- ExternalGameGym                # Gymnasium wrapper
  |     +-- reward_fn                # Milestone / Exploration / Progress / Battle
  |
  +-- TemporalObsWrapper             # +8 delta features (velocity, HP change, stuck, etc.)
  +-- AntiLoopWrapper                # Softened penalties for spam/cycling/idle
  +-- PokemonTelemetryWrapper        # Game event detection, stats for TensorBoard
  +-- MultiInputWrapper              # (with --screen) splits obs -> Dict{screen, ram}
```

Both paths -- the headless Rust engine (CsLite, Tactical) and the external game bridge (Pokemon Gold) -- produce standard `gym.Env` objects. The training pipeline doesn't care which one it gets.

## Files

| File | Purpose |
|------|---------|
| `bridges/profiles/pokemon_gold.py` | 25 RAM addresses, bridge factory, reward functions |
| `bridges/rewards/milestone.py` | Map milestone + count-based exploration + dialogue reward |
| `bridges/wrappers/temporal.py` | Delta features (velocity, HP change, battle/menu transitions) |
| `bridges/wrappers/anti_loop.py` | Spam/cycle/idle penalties |
| `bridges/wrappers/multi_input.py` | Splits flat obs into Dict{screen, ram} for MultiInputPolicy |
| `bridges/wrappers/pokemon_telemetry.py` | Game event detection, cumulative stats, high-water-mark tracking |
| `bridges/wrappers/pokemon_eval.py` | Honest stochastic eval callback with game progress metrics |
| `bridges/wrappers/gold_callbacks.py` | TensorBoard telemetry callback |
| `train_gold.py` | Training + evaluation CLI |
| `save_gold_state.py` | Create save states (close window to save) |
| `test_vlm_pokemon.py` | VLM screenshot interpretation test (Gemma 4) |

## Observation Space

### RAM Features (25)

Sourced from the [pret/pokegold](https://github.com/pret/pokegold) disassembly via the compiled `pokegold.sym` symbol file. All addresses in WRAM bank 1, read via PyBoy's banked memory access (`memory[1, addr]`).

| Group | Features | Addresses | Notes |
|-------|----------|-----------|-------|
| **Position** (6) | player_x, player_y, map_group, map_number, player_direction, facing_direction | 0xDA03, 0xDA02, 0xDA00, 0xDA01, 0xD205, 0xCF2F | Direction: 0/4/8/12 (x4) or 0-3 |
| **Badges** (2) | johto_badges, kanta_badges | 0xD57C, 0xD57D | Bitfields, 8 bits each |
| **Party** (6) | party_size, party1_hp_hi/lo, party1_maxhp_hi/lo, party1_level | 0xDA22, 0xDA4C-4F, 0xDA49 | HP is 2-byte big-endian |
| **Battle** (2) | battle_mode, battle_type | 0xD116, 0xD119 | 0=overworld, 1=wild, 2=trainer |
| **Money** (3) | money_hi, money_mid, money_lo | 0xD573-D575 | 3-byte BCD encoded |
| **Movement** (4) | step_count, player_state, player_action, standing_tile | 0xD9BD, 0xD682, 0xD208, 0xD20B | step_count wraps at 256 |
| **Menu** (2) | textbox_flags, menu_flags | 0xD19C, 0xCEB8 | Non-zero when menus/text active |

All values normalized to [0, 1] by dividing by the feature's max value.

### Temporal Delta Features (+8)

Added by `TemporalObsWrapper`. Computed from consecutive observations:

| Feature | Description |
|---------|-------------|
| delta_x | Position X change since last step |
| delta_y | Position Y change since last step |
| delta_hp | HP (high byte) change since last step |
| battle_entered | 1.0 if just entered battle, else 0.0 |
| battle_exited | 1.0 if just exited battle, else 0.0 |
| stuck_count | Consecutive steps at same position, normalized to [0, 1] |
| menu_entered | 1.0 if menu just opened |
| menu_exited | 1.0 if menu just closed |

### Screen Observations (optional, via --screen)

With `--screen`, PyBoyObservationSource appends a 36x40 grayscale downscaled screen (1440 pixels) to the observation. `MultiInputWrapper` splits this into a Dict observation:

- `screen`: shape (36, 40, 1), float32, values [0, 1]
- `ram`: shape (33,) with temporal deltas or (25,) without, float32, values [0, 1]

SB3's `MultiInputPolicy` applies a CNN to the screen and an MLP to the RAM, then concatenates before the policy head.

### Frame Stacking

With `--frame-stack 4` (RAM-only), the final observation is 33 x 4 = **132 dimensions**. Not compatible with `--screen` (frame stacking is disabled when using screen observations).

## Action Space

`MultiBinary(6)` -- each element is 0 or 1:

| Index | Button |
|-------|--------|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |
| 4 | A |
| 5 | B |

64 possible button combinations per step. Each step advances 24 emulator frames (~0.4s game time).

Start and Select are excluded. With 8 buttons, 95% of steps were trapped in menus from random Start/Select presses. 6 buttons eliminates this entirely but means the agent cannot access menus, items, or Pokemon management.

## Reward Functions

### MilestoneReward (recommended)

Large one-time bonuses for reaching specific maps, plus count-based exploration between milestones.

| Signal | Reward | Trigger |
|--------|--------|---------|
| Reach Route 29 | +50.0 | Map (24, 3) entered, once per episode |
| Tile exploration | 1.0/sqrt(N) | Each tile visit, diminishing returns |
| Level up | +3.0 per level | High-water-mark tracked |
| Badge earned | +20.0 per badge | High-water-mark tracked |
| New party member | +5.0 | High-water-mark tracked |
| Dialogue advance | +0.1 | Press A when textbox is active |

### ExplorationReward

Count-based exploration only. First visit: +2.0, second: +1.41, Nth: +2.0/sqrt(N). No revisit penalty. Anti-bounce prevents A-B oscillation.

### ProgressReward

ExplorationReward (at 1.0 scale) + badge (+20.0), level (+3.0), and party size (+5.0) bonuses. All high-water-mark tracked.

### BattleReward

For battle training from encounter save states. Win: +5.0, HP preserved: +2.0 x (hp/max_hp), faint: -5.0, critical HP: -0.5.

## Training CLI

```
python train_gold.py [options]

Reward:
  --reward {exploration,progress,battle,milestone,none}  (default: exploration)

Environment:
  --state PATH          Save state file to load
  --max-steps N         Steps per episode before truncation (default: 2048)
  --ticks N             Emulator frames per agent step (default: 24)
  --screen              Include downscaled screen pixels in observations
  --envs N              Parallel environments (default: 4)

Wrappers:
  --temporal            Add delta features (velocity, HP change, stuck counter)
  --anti-loop           Penalize repetitive action patterns (softened values)
  --frame-stack N       Stack N frames (default: 1, recommended: 4 for RAM-only)

Training:
  --steps N             Total training steps (default: 100,000)
  --gamma F             Discount factor (default: 0.998, high for Pokemon's long horizon)
  --lr F                Learning rate (default: 1.5e-4)
  --ent-coef F          Entropy coefficient (default: 0.01)

Display:
  --headed              Show PyBoy window for env 0 (auto-sets max-steps to 8192)

Evaluation:
  --eval PATH           Load model and evaluate instead of training
  --episodes N          Number of eval episodes (default: 5)
```

### Recommended Configurations

```bash
# Quick test -- milestone reward, reaches Route 29 in <20K steps
python train_gold.py \
  --state data/pokemon_gold_ready.state \
  --reward milestone \
  --temporal --anti-loop \
  --steps 50000 --envs 1

# Standard training -- milestone reward, RAM-only
python train_gold.py \
  --state data/pokemon_gold_ready.state \
  --reward milestone \
  --temporal --anti-loop --frame-stack 4 \
  --gamma 0.998 --steps 500000 --envs 4

# With screen observations -- CNN+RAM, slower but richer
python train_gold.py \
  --state data/pokemon_gold_ready.state \
  --reward milestone --screen \
  --temporal --anti-loop \
  --gamma 0.998 --steps 500000 --envs 1

# Long exploration run -- no milestones, pure exploration
python train_gold.py \
  --state data/pokemon_gold_ready.state \
  --reward exploration \
  --temporal --anti-loop --frame-stack 4 \
  --gamma 0.998 --steps 2000000 --envs 4

# Battle training (from save state in front of a trainer)
python train_gold.py \
  --state data/gold_trainer_battle.state \
  --reward battle --temporal \
  --max-steps 512 --steps 500000
```

## Telemetry

All metrics logged via `PokemonTelemetryWrapper` and `GoldTelemetryCallback` to TensorBoard:

```
pokemon/tiles_visited      -- unique positions per episode
pokemon/maps_visited       -- unique maps entered per episode
pokemon/exploration_rate   -- tiles_visited / total_steps
pokemon/wild_battles       -- wild encounters per episode
pokemon/trainer_battles    -- trainer battles per episode
pokemon/battles_won        -- battles survived per episode
pokemon/battles_lost       -- faints per episode
pokemon/level_ups          -- levels gained per episode
pokemon/max_level          -- highest level reached
pokemon/badges_earned      -- cumulative badges
pokemon/steps_in_battle    -- steps spent in battle state
pokemon/steps_in_menu      -- steps with textbox/menu active
pokemon/steps_stuck        -- consecutive steps at same position
```

Honest eval metrics (separate from training rollouts) under `eval_honest/*`:

```
eval_honest/unique_tiles
eval_honest/maps_reached
eval_honest/move_pct
eval_honest/any_route_29
eval_honest/battles
eval_honest/total_reward
```

## Save States

The emulator resets to a save state at the start of each episode (~5ms). This skips the title screen, intro, and name selection.

```bash
# Create a save state interactively
python save_gold_state.py
# Play through the title in the PyBoy window, close when ready

# Create for a specific location
python save_gold_state.py --output data/gold_route_29.state
```

A backup state exists at `data/pokemon_gold_ready_backup.state` in case the primary state gets corrupted.

## Debug Tools

### RAM Viewer

Run the game headed with live RAM readout in the terminal:

```bash
python debug_gold_ram.py --state data/pokemon_gold_ready.state
```

Shows decoded values for all 25 features: position, badges, HP, money, battle state, etc.

### Helpers

```python
from bridges.profiles.pokemon_gold import (
    decode_bcd_money,   # 3-byte BCD -> integer
    decode_hp,          # 2-byte big-endian -> integer
    count_badges,       # bitfield -> count
    raw_byte,           # normalized obs value -> raw byte
)
```

## GBC WRAM Banking

Pokemon Gold stores game state in WRAM bank 1 (0xD000-0xDFFF). On Game Boy Color, this region is bank-switched. PyBoy's `memory[addr]` reads the currently-mapped bank, which may not be bank 1 at the moment of reading.

The bridge uses explicit banked reads for correctness:

```python
# Standard read (depends on current bank -- unreliable for GBC)
val = pyboy.memory[0xDA03]

# Banked read (always reads bank 1 -- correct for Gold)
val = pyboy.memory[1, 0xDA03]
```

This is handled automatically by `PyBoyObservationSource` and `PyBoyHost.read_memory()`. Addresses in the 0xC000-0xCFFF range (WRAM bank 0) are always directly accessible and don't need banked reads.

## Performance

Tested on M4 Max 64GB:

| Config | Steps/sec |
|--------|-----------|
| 1 env, RAM-only, headless | ~700 |
| 1 env, CNN+RAM, headless | ~300 |
| 1 env, headed (speed=3) | ~2.5 |
| 4 envs, SubprocVecEnv (estimated) | ~2,500 |

Each step advances 24 emulator frames. At 700 steps/sec with RAM-only, that's 16,800 emulated frames/sec.

## Training Outputs

Runs are saved to `runs/gold_{reward}_{timestamp}/`:

```
runs/gold_milestone_1747012345/
  final_model.zip          # Final trained model
  best/
    best_model.zip         # Best model by eval reward
  checkpoints/
    gold_1000_steps.zip    # Periodic checkpoints
  eval_logs/
    evaluations.npz        # Eval reward history
  honest_eval/
    eval_00010000.json     # Detailed eval at each checkpoint
    eval_00020000.json     # Includes per-episode game stats
  tb/
    PPO_1/                 # TensorBoard logs
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs/gold_milestone_*/tb
```

## ROM Compatibility

Tested with: `Pokemon - Gold Version (USA, Europe) (SGB Enhanced).gbc`

RAM addresses are from the US Gold disassembly. Other versions (Japanese, European non-SGB) may have different offsets. If values read as zero when in-game, the addresses need remapping for your ROM version.

The ROM file goes in `python/data/` and is gitignored. Set `POKEMON_ROM` env var to override the path.
