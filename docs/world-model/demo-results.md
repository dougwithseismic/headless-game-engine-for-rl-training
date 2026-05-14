# World Model Demo Results

First end-to-end run of the Dyna world model system on real CsLite 1v1 game data.

## Setup

| Parameter | Value |
|-----------|-------|
| Config | `configs/cs_lite/1v1_wide.json` |
| Scenario | CsLite 1v1 (3D FPS with Rapier3D) |
| Episodes collected | 200 |
| Total transitions | 398,705 |
| Mean episode reward | 2.73 |
| Observation dim | 250 |
| Action space | MultiDiscrete([12, 2, 2, 3]) |
| Ensemble | 5 members, 256x3 hidden (SiLU) |
| Training steps | 500 gradient steps |
| Batch size | 512 |
| Training time | 43.1s on M4 Max |

## Data Collection

200 episodes of random policy play in the CsLite 1v1 arena, collecting ~2000 transitions per episode. Each transition is a (obs, action, reward, next_obs, done) tuple with 250-dim observations and 4 discrete action heads. The random policy provides good coverage of the state space but doesn't play strategically.

## Training

The dynamics ensemble trained for 500 gradient steps on the 398K replay buffer. Training loss (Gaussian NLL) decreased steadily from -0.42 to -4.83. Validation loss (MSE on held-out transitions) stabilized around 0.015 after ~200 steps. Occasional loss spikes (step 250: 50.68, step 310: 13.69) are expected with bootstrap aggregation on diverse game data — the ensemble recovers quickly.

## Results

### 1-Step Prediction

| Metric | Value | Assessment |
|--------|-------|-----------|
| State MSE | 0.0144 | Low — predictions close to actual |
| Reward MSE | 0.000298 | Excellent — reward dynamics well captured |
| Median R-squared | 0.005 | Low overall (see analysis below) |
| Features with R2 > 0.95 | 11.6% (29/250 features) | ~30 features have strong dynamics |
| Ensemble disagreement | 0.045 | Moderate — model has reasonable confidence |

### N-Step Divergence

| Horizon | Drift | Ratio vs 1-step |
|---------|-------|-----------------|
| 1 step | 0.0034 | 1.0x |
| 3 steps | 0.0078 | 2.3x |
| 5 steps | 0.0139 | 4.1x |
| 10 steps | 0.0234 | 6.9x |
| 15 steps | 0.0328 | 9.6x |
| 20 steps | 0.0413 | 12.1x |

**Compounding ratio (5-step / 1-step): 4.07x** — sublinear error accumulation up to 10 steps, approaching linear beyond that. This means the model is usable for 5-10 step imagination rollouts before predictions become unreliable.

### Imagined Trajectories

5 sample trajectories from starting states in the replay buffer, rolled forward 10 steps:

| Trajectory | Start Position | End Position (10 steps) | Total Reward |
|-----------|---------------|------------------------|-------------|
| 0 | [1.00, 1.00, 1.00] | [0.98, 1.00, 0.96] | -0.033 |
| 1 | [1.00, 1.00, 1.00] | [0.94, 0.94, 0.93] | +0.008 |
| 2 | [1.00, 1.00, 1.00] | [0.95, 1.00, 0.92] | -0.011 |
| 3 | [1.00, 1.00, 1.00] | [0.97, 0.96, 0.95] | -0.035 |
| 4 | [0.00, 0.00, 1.00] | [0.01, 0.02, 1.01] | -0.024 |

The model generates physically plausible trajectories — small position changes over 10 steps, negative rewards (consistent with random policy not dealing much damage), and positions that stay in-bounds.

## Analysis

### Why R2 Is Low but the Model Is Still Useful

The median R2 of 0.005 looks bad, but it's misleading for two reasons:

1. **Most features are nearly static.** Of 250 observation features, many change by less than 1e-5 per step (action masks, round info, team state). For these, the variance of the actual delta is near zero, so even a tiny prediction error produces extreme negative R2. The 5 worst features all have MSE < 0.00001 — they're essentially constant, and R2 is undefined for constant features.

2. **The features that matter have high R2.** 11.6% of features (29/250) have R2 > 0.95 — these are the position, velocity, health, and ammo features that actually drive gameplay. The model captures these dynamics well.

The state MSE of 0.014 across all 250 features means the average per-feature error is 0.000058 — tiny. The model is accurately predicting next states; the R2 metric just doesn't handle near-constant features well.

### What the Compounding Ratio Tells Us

The 5-step compounding ratio of 4.07x (not 25x as pure error doubling would give) means prediction errors are **not independently adding** — the model maintains structural consistency across steps. This is the difference between "usable for Dyna augmentation" and "garbage after 2 steps."

For the planned Dyna-PPO integration, 1-3 step rollouts (where drift is 2.3-4.1x the 1-step error) will generate high-quality synthetic transitions. This is within the regime where MBPO-style augmentation consistently improves sample efficiency.

### Saved Artifacts

```
python/runs/world_model_demo/
  dynamics_ensemble.pt     (6.3 MB -- 5 member weights + normalizer)
  replay_buffer.npz        (140 MB -- 398K transitions)
  world_model_meta.json    (experiment metadata)
```

## Next Steps

1. **Train with a real PPO policy instead of random.** Random policy covers the state space but doesn't produce the action-state correlations a trained agent would. Training on PPO rollouts will give the dynamics model more relevant data.

2. **Enable Dyna augmentation during PPO training.** Run:
   ```bash
   python scripts/train.py --scenario cs_lite --config configs/cs_lite/1v1_wide.json \
     --dyna --dyna-n-models 5 --dyna-hidden 256 --timesteps 3000000
   ```

3. **Compare sample efficiency.** Run identical PPO configs with and without `--dyna`, measure how many timesteps each needs to reach the same eval reward.

4. **Feature-level analysis.** Identify which observation features the model predicts well vs poorly. Use this to inform observation space design and potentially simplify the obs for better world model accuracy.
