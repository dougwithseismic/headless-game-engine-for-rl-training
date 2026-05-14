# MPC Experiment Results: Does Looking Ahead Help?

## The Question

We have a dynamics model that accurately predicts CsLite combat transitions. Can we use it to make a trained agent play better by planning ahead at each decision step?

## Setup

- **Policy**: `search_heavy_symmetric` (reward ~280 vs scripted AI, ~118 in self-play)
- **Dynamics model**: 5-member ensemble trained on 423K expert transitions (val_loss 0.30)
- **Config**: `1v1_wide.json`, CsLite 1v1
- **MPC**: CEM planner, 256 candidates, 32 elites, 2 iterations, horizon 5
- **Episodes**: 50 per condition

## Results

| Condition | Mean | Std | Min | Max | vs RAW |
|-----------|------|-----|-----|-----|--------|
| **RAW** (baseline) | **118.5** | 1.5 | 117.0 | 122.8 | --- |
| **TACTICAL** (policy=move, MPC=combat) | 3.9 | 1.3 | 2.2 | 12.3 | -114.6 |
| **GATED** (override only if MPC >> policy) | 118.2 | 4.5 | 97.0 | 124.5 | -0.3 |
| **FULL MPC** (MPC controls everything) | 6.3 | 5.5 | 2.5 | 31.2 | -112.3 |

**MPC planning made things catastrophically worse.**

FULL MPC and TACTICAL both dropped reward from 118 to 4-6 — a 97% decrease. GATED was identical to raw policy because the override gate barely opened (2.6% of steps).

## Why MPC Failed

### The Core Problem: Flat Reward Landscape

The dynamics model predicts per-step reward, which in CsLite averages ~0.01 per step with very little variance between actions. A round reward of ~118 is accumulated over ~1400 steps, so each step contributes ~0.08 on average. The difference between a good and bad action at any single step is typically 0.001-0.003.

MPC finds marginal improvements in per-step predicted reward (0.012 vs 0.013) but picks completely different actions to get there. These "improvements" are within the model's prediction noise — the planner is optimizing noise.

### What Went Wrong Specifically

**FULL MPC**: The planner picked incoherent movement — jumping between random compass directions every step because each one predicted nearly identical reward. The policy moves consistently (action 10 "advance" for 20+ steps); MPC picked actions 0, 1, 2, 3, 4, 6, 8 in consecutive steps. Coherent movement is essential for FPS gameplay but invisible to per-step reward.

**TACTICAL MPC**: The planner chose to shoot constantly (action head 1 = 1) because shooting correlated with reward in the training data. But the expert agent only shot when aimed at the enemy. The model learned "shoot → reward" (correlation) not "shoot when aimed → reward" (causation). MPC doesn't understand aim.

**GATED MPC**: The 0.05 threshold was almost never exceeded (only 2.6% of steps) because predicted rewards are so similar between actions. When it did override, the replacement action was no better than random — same flat reward landscape problem.

## What This Tells Us

### MPC with observation-space models doesn't work for FPS

The dynamics model predicts in raw observation space (250 features). In this space, the reward landscape is too flat for planning — small differences between actions are lost in prediction noise. This is a known limitation: observation-space models capture state transitions accurately but don't provide a useful gradient for planning.

**What would work**: a model that predicts in **latent space** where the reward landscape has more structure. TD-MPC2 does exactly this — it learns a compressed representation where planning is meaningful. Dreamer's RSSM provides this too.

### This doesn't mean Dreamer will fail

MPC and Dreamer use the world model completely differently:

- **MPC asks**: "Model, what should I do right now?" → The model says "all actions look the same" → Agent acts randomly → Catastrophe
- **Dreamer asks**: "Model, give me 15 steps of imagined experience" → The actor-critic learns from those steps just like it would from real steps → Policy gradually improves → Works fine

Dreamer doesn't need the model to distinguish good from bad actions at each step. It needs the model to produce realistic-enough trajectories for the actor-critic to learn from. Our model does that — state MSE is low, reward prediction is accurate, compounding ratio is 4x at 5-step. The trajectories would be realistic enough for training.

### The real path to gun cleric mode

1. **Dreamer** for training: the policy internalizes "thinking ahead" through millions of imagined trajectories during training. At inference it's reactive but the intelligence is baked in.

2. **Latent-space MPC** (TD-MPC2 style) for inference: if we want real-time planning, we need a latent world model where the reward landscape has structure. This means training a compact latent dynamics model (not our current 250-dim observation-space ensemble) with a Q-function that provides meaningful action evaluation.

3. **Hierarchical planning**: MPC at the observation level fails because step-level rewards are flat. But at the skill level ("should I peek?" "should I commit?"), the reward differences are large. Planning at an abstract action level, not the tick level, could work.

## Decision Time Measurements

| Condition | Per-step | Effective Hz |
|-----------|----------|-------------|
| RAW | <0.1ms | 10,000+ Hz |
| MPC (CEM) | 12ms | ~83 Hz |

MPC at 12ms per step is fast enough for real-time use (83 Hz > 60 Hz game tick). The speed isn't the problem — the planning quality is.

## Artifacts

```
python/scripts/eval_mpc.py           # Evaluation script (4-way comparison)
python/training/mpc_planner.py       # MPC planner (random shooting + CEM)
```
