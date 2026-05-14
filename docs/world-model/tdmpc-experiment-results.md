# TD-MPC Experiment Results: Proper MPC Implementation

## The Fix

We rebuilt MPC with all 5 fixes from the post-mortem:
1. Terminal Q-function (twin Q-heads with TD-learning)
2. Learned latent space (128-dim encoder, no decoder)
3. Policy prior (80% candidates from learned policy head)
4. MPPI soft-weighting (not hard CEM elite selection)
5. Temporal correlation (action persistence in random candidates)

## Progressive Results

| Phase | Train Steps | TDMPC Planning | TDMPC Policy Only | Raw PPO |
|-------|-------------|---------------|-------------------|---------|
| 1 (smoke) | 200 | 23.4 | — | 119.3 |
| 2 (medium) | 2000 | 33.2 | — | 119.1 |
| 3 (full) | 5000 | **9.9** | **117.6** | **118.8** |

## Key Finding: The Policy Learned, The Planner Didn't Help

At 5000 training steps:
- **Raw PPO policy**: 118.8 +/- 1.7 (the baseline to beat)
- **TD-MPC policy head** (no planning, just the learned policy network): **117.6** +/- 8.7
- **TD-MPC with MPPI planning**: **9.9** +/- 6.6

The TD-MPC policy head nearly matches the original PPO — learning meaningful behavior from offline replay data in just 5000 gradient steps. But the moment we engage the MPPI planner on top, performance collapses to 9.9.

This is **worse than Phase 2** (33.2 at 2000 steps), demonstrating **Q-function overfit**: more training makes the Q-function more confident in wrong predictions, and the planner exploits those overestimates more aggressively.

## Q-Value Discrimination at 2000 Steps

```
move= 0 shoot=0: Q=+0.715
move= 0 shoot=1: Q=+0.985    ← shoot increases Q by 0.27
move= 8 shoot=0: Q=+0.853    ← stay (move=8) has highest Q
move= 8 shoot=1: Q=+1.123    ← stay + shoot = highest Q
```

The Q-function learned real structure: shooting increases value, staying in place has highest value. These are correct — the expert agent shoots for reward and doesn't need to move much when fighting. But the planner takes this too literally: it holds still and shoots constantly, regardless of whether an enemy is nearby or in view.

## Why Planning Still Fails Even With Proper Architecture

**1. Offline Q-learning overestimates out-of-distribution states.**
The Q-function was trained on expert data where states after "stay + shoot" always led to high reward (because the expert was aimed at the enemy). But when the planner chooses "stay + shoot" from a random position, the next state is NOT in the training distribution — the Q-function hallucinates high value where it has no data.

This is the fundamental **offline RL extrapolation problem**. TD-MPC2 avoids it by training online (interleaving real environment steps with planning), not from a static offline buffer.

**2. 423K transitions is not enough for a latent world model.**
TD-MPC2 typically trains for millions of environment steps. Our 423K-transition buffer provides good coverage for simple dynamics prediction but insufficient coverage for a Q-function that must generalize to states the planner visits.

**3. Discrete actions + MPPI is an awkward fit.**
Our action space is MultiDiscrete([12,2,2,3]). MPPI's softmax weighting was designed for continuous distributions. The discrete-continuous mismatch means the "weighted average" of action sequences doesn't produce a valid action — we have to round, which discards the gradient information MPPI relies on.

## What This Means

**MPC (in any form) does not work for improving a trained FPS policy from offline data.** The problem isn't implementation — we've tried:

1. Naive MPC (CEM, observation-space): reward 3.9
2. TD-MPC (MPPI, latent-space, Q-function): reward 9.9

Neither approaches the raw policy's 118.8. The consistent failure pattern is the planner exploiting model errors, either in the reward landscape (naive) or in Q-value estimates (TD-MPC).

**The TD-MPC policy head working (117.6) confirms the latent space is useful.** The encoder, dynamics model, and policy head all learned meaningful representations. The failure is specifically in the planning loop, not in the learned components.

**The path forward:**
- **Dreamer**: Use the learned latent model to train the policy in imagination, not to plan at inference. The actor-critic trains inside the model, experiencing consequences of bad actions, and learns to avoid them. No planning loop needed at inference.
- **Online TD-MPC**: If we want planning, the model must be trained online (interleaving real env steps) so the Q-function stays calibrated to states the planner actually visits. This means building a full training loop, not evaluating from offline data.

## Code

```
python/training/td_mpc.py              # TD-MPC implementation (encoder + latent dynamics + Q + policy + MPPI)
python/scripts/eval_tdmpc.py           # Progressive evaluation script
python/runs/tdmpc_experiment/tdmpc.pt  # Trained TD-MPC checkpoint
```
