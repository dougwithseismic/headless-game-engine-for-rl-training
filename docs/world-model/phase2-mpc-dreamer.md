# Phase 2: MPC + Dreamer — The Gun Cleric Architecture

Phase 1 proved the dynamics are learnable. Phase 2 is about using that knowledge to make agents that think ahead.

Two approaches, complementary not competing:

| | MPC (online planning) | Dreamer (imagination training) |
|---|---|---|
| **When it thinks** | At each decision step during play | During training (millions of imagined trajectories) |
| **Inference speed** | 1-5ms per action (256 candidates, 5-step horizon) | Sub-millisecond (single forward pass) |
| **What it's good for** | Critical moments: should I peek? commit? rotate? | Everything: aim, movement, all mechanics |
| **Analogy** | Chess player calculating 5 moves ahead mid-game | Chess grandmaster who's studied so many positions the right move is instinct |

The gun cleric is both: trained in imagination (Dreamer) so the base policy is excellent, with the ability to plan ahead (MPC) when the stakes are high.

## MPC: Planning at Each Step

### How It Works

Given the current observation, the MPC planner:
1. Samples N candidate action sequences (e.g., 256 random 5-step plans)
2. Rolls each forward through the dynamics model
3. Scores by cumulative discounted predicted reward
4. Returns the first action of the best sequence

**Already built** in `python/training/mpc_planner.py`. Two strategies:
- **Random shooting**: sample uniformly, pick best. 0.7ms per plan.
- **CEM (Cross-Entropy Method)**: iteratively refine toward best sequences. 4ms per plan. Improves over random 84% of the time on synthetic data.

### What the Literature Says

**TD-MPC2** (Hansen et al., ICLR 2024) is the state of the art for MPC in RL:
- Learns an implicit world model entirely in latent space (no decoder)
- Plans via MPPI (512 samples x 6 iterations x H=5) in latent space
- Terminal Q-value handles long-term credit assignment beyond the planning horizon
- Single 317M parameter model handles 80 tasks across multiple domains
- Key insight: the **policy prior** seeds the planner with good initial action sequences, so planning is a refinement rather than search from scratch

**Dream-MPC** (2025) combines Dreamer's RSSM with gradient-based planning:
- Only 15 model evaluations (vs MPPI's 9,216) by using backprop through the differentiable world model
- Matches or exceeds MPPI performance
- The most compute-efficient planning approach in the literature

### Practical Limits for FPS

At 60Hz (16.7ms budget per decision):
- **H=5 at our tick rate** covers ~250ms of lookahead — enough for peeking, micro-positioning, fight commitment
- **Not enough for** trades (0.5-2s), rotations (5-30s), strategy (minutes)
- Planning assumes opponent is ~stationary during the horizon — valid for 250ms, not for seconds

### Where MPC Fits in Our Stack

MPC is most valuable at the **skills layer** of the hierarchical framework:
- The RL motor (aim, strafe, recoil) runs reactively at full tick rate — planning adds nothing here
- The skills layer (peek timing, fight/flee decisions, utility usage) benefits from 3-5 step lookahead
- The strategy layer uses abstract planning (MCTS at option level, or LLM) — seconds/minutes horizon

## Dreamer: Training in Imagination

### How It Works

Instead of training the policy on real environment steps (like PPO), Dreamer:
1. Collects real experience into a replay buffer
2. Trains the RSSM world model on replay data
3. **Imagines** 15-step trajectories from encoded real states — no env interaction needed
4. Trains actor-critic entirely on imagined trajectories using backprop through the model
5. At inference: single forward pass through encoder + GRU + actor. **No planning, no search.**

The policy at test time is purely reactive — all the "thinking ahead" happened during training via millions of imagined trajectories. This is the grandmaster: trained on so many positions that the right response is instinct.

### Why Dreamer Matters

- **Sample efficiency**: DreamerV3 is 10-100x more sample-efficient than PPO. It reached diamond in Minecraft from scratch — something no model-free method has done.
- **The policy is fast at inference**: sub-millisecond, no search overhead. Runs at full tick rate.
- **It scales**: DreamerV3 masters 150+ tasks with a single hyperparameter set. Published in Nature (2025).

### DreamerV3 Architecture (for state-vector observations)

Since GhostLobby uses state vectors (not pixels), the architecture simplifies:

```
Encoder:     MLP (obs_dim -> 2x1024) -> latent posterior z_t
Sequence:    GRU (4096 units) -> deterministic state h_t
Dynamics:    MLP -> prior p(z_t | h_t) (predict z without observation)
Decoder:     MLP -> reconstruct observation (for training only)
Reward:      MLP -> predict reward (twohot symlog)
Continue:    MLP -> predict episode termination (Bernoulli)
Actor:       MLP (4x1024) -> action distribution
Critic:      MLP (4x1024) -> value estimate (twohot symlog)
```

Key: the stochastic state z_t is 32 categorical distributions x 32 classes = 1024 discrete latent dimensions. This captures multi-modal futures (opponent might go left OR right).

### What Dreamer Doesn't Do (Yet) for FPS

- **No adversarial results**: DreamerV3/V4 benchmarks are all single-agent. No published results on competitive games.
- **No opponent modeling**: the world model learns environment dynamics but doesn't explicitly model opponent decision-making.
- **No multi-timescale planning**: the imagination horizon (15 steps) is fixed, not adapted to decision importance.

These are all research opportunities for GhostLobby.

## The Hybrid: Dreamer Training + MPC Inference

The emerging best practice from the literature:

1. **Train with Dreamer** for sample efficiency — the policy learns excellent base behavior from millions of imagined trajectories
2. **Deploy the reactive policy** for 95%+ of decisions — it's fast and good enough
3. **Optionally engage MPC** for critical moments — the dynamics model is already trained, just add a planning loop

This is what TD-MPC does: the **policy prior** (Dreamer-trained base policy) seeds the planner with good initial trajectories, and MPPI planning refines them. The planning doesn't search from scratch — it improves on what the policy would already do.

**For GhostLobby's hierarchical architecture:**

| Layer | Training | Inference | Speed |
|-------|----------|-----------|-------|
| RL Motor (aim, move) | Dreamer imagination | Reactive policy | Full tick rate |
| Skills (peek, fight) | Dreamer + skill conditioning | Reactive OR MPC for critical decisions | 20-60Hz |
| Strategy (rotate, buy) | LLM + world model simulation | MCTS at abstract action level | 0.1-0.5Hz |

## Implementation Plan

### Step 1: Integrate DreamerV3 with GhostLobbyEnv

Fork `NM512/dreamerv3-torch` (cleanest PyTorch port). Adapt for our env:
- Register `CsLiteGym` as a Dreamer environment
- Map MultiDiscrete action space to Dreamer's action handling
- State-vector observations → MLP encoder (replace CNN)
- Configure: 15-step imagination, 512 train ratio, XL model (1024 dim)

### Step 2: Train Dreamer Agent

- Run DreamerV3 on CsLite 1v1 with self-play
- Compare sample efficiency vs PPO baseline (same total env steps)
- Evaluate: does Dreamer reach the same reward in fewer steps?

### Step 3: Add MPC Planning Layer

- Use the Dreamer-trained RSSM as the dynamics model for MPC
- Implement TD-MPC-style MPPI: policy prior + short-horizon planning
- Measure: does planning improve over the base policy on specific mechanics (peek timing, fight commitment)?

### Step 4: Evaluate Gun Cleric Behavior

The real test: does the agent exhibit **predictive** behavior?
- Pre-aims at positions enemies haven't reached yet
- Times peeks for when the opponent is likely reloading
- Commits to fights only when the model predicts favorable outcomes
- Holds position when the model predicts unfavorable trades

## Key References

- TD-MPC2: Hansen et al. (ICLR 2024) — [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)
- DreamerV3: Hafner et al. (Nature 2025) — [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Dream-MPC (2025) — [dream-mpc.github.io](https://dream-mpc.github.io/)
- EfficientZero V2: Wang et al. (ICML 2024) — [arXiv:2403.00564](https://arxiv.org/abs/2403.00564)
- Dreamer V4: Hafner & Yan (2025) — [arXiv:2509.24527](https://arxiv.org/abs/2509.24527)

## Repos

- DreamerV3 PyTorch: [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
- TD-MPC2: [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2)
- LightZero (MuZero family): [opendilab/LightZero](https://github.com/opendilab/LightZero)
- SheepRL (multi-algo framework): [Eclectic-Sheep/sheeprl](https://github.com/Eclectic-Sheep/sheeprl)
