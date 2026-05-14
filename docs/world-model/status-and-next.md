# World Model Research: Final Status & Next Steps

## What Was Accomplished

### Phase 1: Dynamics Model (COMPLETE, VALIDATED)
- MLP ensemble dynamics model that learns FPS combat transitions in 43s
- Replay buffer that stabilizes self-play training (collapse rate 30% → 10%)
- Integrated into PPO training pipeline via `--dyna` flag
- Validated across 5 experiments with real game data

### Phase 2: MPC Planning (COMPLETE, NEGATIVE RESULT)
- Naive MPC (CEM): reward 3.9 vs baseline 118 — flat reward landscape
- TD-MPC v1 (latent + Q-function + MPPI): reward 9.9 — Q-function overfit
- TD-MPC v2 (all paper fixes): built but not producing competitive results
- **Key finding**: MPC from offline data doesn't work for FPS. Online training required.
- **Key finding**: MCTS (not MPC) is the right tool for turn-based games like Pokemon

### Phase 3: Dreamer + Online TD-MPC (BUILT, NOT YET FUNCTIONAL)
- DreamerV3-lite: RSSM, categorical latents, imagination actor-critic, online training loop
- TD-MPC v2: SimNorm, multi-step training, Q-maximization, MPPI with policy prior
- Online training script: both algorithms with real env interaction
- **Result**: ~3 reward after 50K online steps (PPO baseline: 118)
- **Root cause**: Model too small + too few steps + from-scratch implementation missing critical details

## Why From-Scratch Didn't Work

DreamerV3 paper uses:
- 200M parameters (we used ~2M)
- 100M+ env steps (we used 50K)
- 512 gradient steps per env step (we used 1-16)
- Years of implementation refinement

Our code is architecturally correct but ~100x underscaled.

## What To Do Next

### Option A: Fork dreamerv3-torch (RECOMMENDED)
```bash
git clone https://github.com/NM512/dreamerv3-torch
# Register CsLiteGym as environment
# Configure for state-vector observations (MLP encoder, not CNN)
# Run with paper-scale defaults on the 3090 desktop
```
This gets paper-quality DreamerV3 with proper scale. Estimated integration: 1-2 days. Training: hours on 3090.

### Option B: Fork SheepRL
Multi-algorithm framework with DreamerV3 + distributed training. More setup but supports multi-GPU.

### Option C: Fork TD-MPC2
Best for continuous control. Needs adaptation for discrete actions (MultiDiscrete).

### Hardware Recommendation
- **3090 desktop**: Run DreamerV3 at full scale. 24GB VRAM handles 200M params easily.
- **M4 Max**: Good for PPO (CPU-bound, parallel envs). Not ideal for world model training (GPU-bound, needs CUDA for speed).

## Code Inventory

```
python/training/
  replay_buffer.py          # FIFO buffer — WORKING, VALIDATED
  dynamics_model.py         # MLP ensemble — WORKING, VALIDATED
  dynamics_trainer.py       # Training orchestrator — WORKING
  dynamics_eval.py          # Evaluation suite — WORKING
  dyna_callback.py          # SB3 PPO integration — WORKING, VALIDATED in A/B test
  mpc_planner.py            # Naive MPC (CEM) — WORKING, but MPC approach failed
  td_mpc.py                 # TD-MPC v1 — WORKING, Q-function overfit issue
  td_mpc_v2.py              # TD-MPC v2 (audited fixes) — BUILT, undertrained
  dreamer.py                # DreamerV3-lite — BUILT, undertrained

python/scripts/
  demo_world_model.py       # End-to-end demo — WORKING
  eval_mpc.py               # MPC evaluation — WORKING
  eval_tdmpc.py             # TD-MPC evaluation — WORKING
  eval_dreamer.py           # Dreamer evaluation — WORKING
  analyze_ab_test.py        # A/B analysis — WORKING
  train_online_worldmodel.py # Online training loop — BUILT

docs/world-model/
  index.md                  # Reading guide
  direction.md              # Full research landscape
  demo-results.md           # First demo on real data
  ab-test-results.md        # PPO vs Dyna A/B
  3way-ab-test-results.md   # 3-way with reward shaping
  phase2-mpc-dreamer.md     # MPC + Dreamer research
  mpc-experiment-results.md # MPC experiment (failed)
  mpc-when-it-works.md      # When to use MPC vs MCTS
  tdmpc-experiment-results.md # TD-MPC experiment
  online-training-results.md  # Online training results
  status-and-next.md        # This document
```

## Research Findings Summary

1. **FPS dynamics are learnable** from state vectors (R2 > 0.95, 43s training)
2. **Replay buffer stabilizes self-play** (30% → 10% collapse rate, peak 120→124)
3. **MPC doesn't work for FPS** (flat reward landscape, planner optimizes noise)
4. **MCTS is right for turn-based games** (Pokemon: Foul Play bot got 90%+ win rate)
5. **Offline world model training doesn't transfer** to real gameplay
6. **Online training is required** but needs paper-scale compute
7. **The path forward is forking battle-tested repos**, not building from scratch
