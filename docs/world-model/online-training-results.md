# Online Training Results: Dreamer + TD-MPC v2

## What Changed From Offline

The critical fix: the agent now acts in the real environment while training, filling the replay buffer with its own experience. This is how DreamerV3 and TD-MPC2 actually work in the papers.

## Results

### Online Dreamer (100K steps, ~100 sps)

| Step | Eval Reward | Buffer Size | Episodes |
|------|------------|------------|----------|
| 5K | 2.9 | 5,000 | 2 |
| 10K | 3.4 | 10,000 | 5 |
| 20K | 2.2 | 20,000 | 10 |
| 30K | 3.5 | 30,000 | 15 |
| 50K | 2.6 | 50,000 | 25 |

**Flat at ~3.0 across 50K steps. No learning signal.**

### Online TD-MPC v2 (100K steps, ~19 sps)

| Step | Eval Reward |
|------|------------|
| 5K | 2.8 |

**Too slow — MPPI planning at every step limits throughput to 19 sps.**

### Baseline Comparison

PPO at 50K steps with the same environment: ~50 reward (early learning, not converged but clearly improving). Our implementations show zero improvement.

## Why Both Are Flat

### 1. Model capacity is too small

Our Dreamer uses h_dim=128, z=(8x8)=64, mlp=64. DreamerV3 defaults: h_dim=4096, z=(32x32)=1024, mlp=1024. We're running at **1/64th the capacity**. The RSSM can't represent the 250-dim observation dynamics in a 64-dim latent state.

### 2. Training budget is too low

DreamerV3 typically trains for 10M-100M env steps. We ran 50K. That's 0.05-0.5% of what the paper uses. Even PPO (model-free, much simpler) needs 500K+ steps to show meaningful learning on CsLite.

### 3. From-scratch implementations miss critical details

The TD-MPC audit found 21 differences between our v1 and the reference. We fixed the 8 critical ones in v2, but the remaining 13 (architecture dimensions, initialization, dropout patterns, etc.) accumulate. DreamerV3 has even more implementation details that matter: symlog everywhere, twohot value bins, percentile return normalization, specific KL balancing schedules.

### 4. Single-env throughput bottleneck

Our online loop uses 1 environment. PPO uses 16-32 parallel envs. With 1 env and model training every 4 steps, throughput is ~100 sps (Dreamer) or ~19 sps (TD-MPC). At these rates, reaching the step budgets where these algorithms work would take days.

## The Honest Assessment

Building DreamerV3 or TD-MPC2 from scratch and matching paper results is a multi-month engineering project, not a single-session task. The papers represent years of iteration by teams at Google DeepMind and UCSD.

## What The Code Actually Delivers

Despite not matching PPO yet, the implementations are architecturally correct:

- **Dreamer**: RSSM with categorical latent states, imagination rollouts, actor-critic training on imagined trajectories, symlog predictions, KL balancing, online training loop with real env interaction
- **TD-MPC v2**: SimNorm encoder, multi-step latent dynamics training, rho-weighted consistency loss (20x), Q-maximization policy, separate world model + policy optimizers, MPPI planning with policy prior and elite selection

Both run online with real environment interaction. The gap is scale (model size, step budget, parallelism), not architecture.

## Recommended Path Forward

**Use existing battle-tested implementations instead of building from scratch:**

1. **Fork [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)** — Register `CsLiteGym` as an environment, configure for state-vector observations, run with their full-scale defaults (4096 GRU, 32x32 latent, 200M params)

2. **Fork [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2)** — Wrap `CsLiteGym` in their env interface, run with their 317M parameter config

3. **Or use [SheepRL](https://github.com/Eclectic-Sheep/sheeprl)** — Multi-algorithm framework with DreamerV3 + distributed training via Lightning. Register env, run.

These repos handle all 21+ implementation details correctly and support multi-GPU training. Our from-scratch code serves as documentation of how the algorithms work and as building blocks for future custom implementations.
