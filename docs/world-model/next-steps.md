# World Model: Next Steps

Status: Phase 1 Dyna system built, demo'd on 400K real transitions from random policy. All code integrated into PPOTrainer with `--dyna` flag. Need to validate on trained agent data and measure actual training speedup.

## Experiment 1: Train World Model on Skilled Agent Data

The demo used random policy data. Trained agents produce much richer data — intentional combat, tactical movement, aim patterns. The dynamics model will learn sharper predictions from this.

```bash
# Collect 500 episodes from the phase3 self-play model (best: reward 250)
cd /Users/godzillaaa/Documents/WEB_PROJECTS/clients/headless-fps-engine

.venv/bin/python python/scripts/demo_world_model.py \
  --config configs/cs_lite/phase3_selfplay.json \
  --scenario cs_lite \
  --episodes 500 \
  --train-steps 1000 \
  --n-models 5 \
  --hidden 256 \
  --n-layers 3 \
  --batch-size 512 \
  --save-dir python/runs/world_model_trained_agent
```

But first we need to modify `demo_world_model.py` to accept a `--policy` flag that loads a trained model and uses it instead of random actions. This produces expert-quality transitions.

**What to measure:**
- Compare R2 scores: random policy data vs trained agent data
- Compare reward prediction accuracy
- Compare N-step divergence (hypothesis: lower compounding ratio with coherent data)

## Experiment 2: A/B Test — PPO vs PPO+Dyna

The key experiment. Same config, same timesteps, one with `--dyna`, one without. Measure which reaches the same eval reward first.

```bash
# Baseline: standard PPO (from phase2 best model)
.venv/bin/python python/scripts/train.py \
  --scenario cs_lite \
  --config configs/cs_lite/phase3_selfplay.json \
  --resume python/runs/cs_lite_phase2_final_20260511_185540/best/best_model.zip \
  --self-play --timesteps 5000000 \
  --name phase3_baseline

# Dyna: PPO + world model augmentation (same starting point)
.venv/bin/python python/scripts/train.py \
  --scenario cs_lite \
  --config configs/cs_lite/phase3_selfplay.json \
  --resume python/runs/cs_lite_phase2_final_20260511_185540/best/best_model.zip \
  --self-play --timesteps 5000000 \
  --dyna --dyna-n-models 5 --dyna-hidden 256 \
  --name phase3_dyna
```

**What to measure:**
- Eval reward curves over timesteps (TensorBoard overlay)
- Timesteps to reach reward threshold (e.g., 200, 225, 250)
- Final reward at same timestep budget
- Behavior metrics: accuracy, K/D, win rate

**Expected:** Dyna run reaches same reward 2-5x faster in timesteps. May also reach a higher ceiling if synthetic data exposes the agent to rare combat situations.

## Experiment 3: Collect Expert Demos via Trained Policy

Before running the A/B test, we need a script change to collect transitions from a trained model instead of random policy.

**What to build:**
- Add `--policy` flag to `demo_world_model.py` that loads a trained .zip model
- Collect transitions using that policy's actions instead of `env.action_space.sample()`
- This produces "expert" replay data for the world model

## Experiment 4: World Model Quality Dashboard

Add TensorBoard logging during Dyna-augmented PPO training so we can watch the world model improve alongside the policy.

**Already built** — the DynaCallback logs `dynamics/state_mse`, `dynamics/reward_mse`, `dynamics/median_r_squared`, `dynamics/buffer_size`, etc. to TensorBoard. Just need to run a Dyna training run and open TensorBoard to see it.

## Implementation Order

1. **Add `--policy` flag to demo script** — 30 min. Enables collecting expert data.
2. **Run Experiment 1** — Collect from trained agent, compare world model quality. ~1 hour.
3. **Run Experiment 2** — A/B test. ~2-4 hours per run (5M steps each). This is the money experiment.
4. **Analyze results** — TensorBoard comparison, write up findings.

## What Success Looks Like

- World model R2 > 0.9 on position/velocity/health features with trained agent data
- Dyna run reaches reward 200 in ≤60% of the timesteps that baseline needs
- Compounding ratio stays below 5x at 5-step horizon with trained agent data
- No model exploitation (imagined reward doesn't diverge from real reward)

## If It Doesn't Work

- **Model exploitation:** Agent games the dynamics model, imagined reward climbs but real reward drops. Fix: shorter rollouts (1 step only), ensemble disagreement penalty.
- **No speedup:** Dyna overhead (model training) eats the sample efficiency gain. Fix: reduce `dyna-train-freq`, smaller ensemble (3 instead of 5).
- **PPO incompatibility:** On-policy PPO rejects off-policy synthetic data. Fix: keep synthetic ratio at 10%, or switch the base algorithm to SAC (off-policy, naturally compatible with Dyna).
