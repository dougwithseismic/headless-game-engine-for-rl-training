# MPC: When It Works, When It Doesn't, and What We Got Wrong

## Did We Implement MPC Incorrectly?

**Yes.** Not a bug — an architectural gap. The research is clear on what we were missing.

### What We Got Wrong

**1. No terminal value function (the biggest issue)**

Our planner scored trajectories by summing predicted rewards over 5 steps. Each step contributes ~0.01 reward with ~0.001 variance between actions. Over 5 steps the total signal is ~0.005 and the noise is larger. The planner is ranking by noise.

TD-MPC/TD-MPC2 appends a learned Q-value at the horizon: `score = sum(rewards) + Q(z_H)`. The Q-function compresses the entire infinite future into one number trained via TD-learning to have large, discriminative differences between good and bad states. This converts a flat 5-step reward landscape into a steep one at the boundary. **This single addition is likely the difference between reward 4 and reward 118.**

**2. Planning in observation space instead of latent space**

Our 250-dim observation-space model wastes capacity predicting near-constant features (action masks, round info) while the few features that matter (position, aim, health) are lost in noise. TD-MPC2 learns a ~256-dim latent space that discards irrelevant features and organizes by task relevance. MuZero's analysis paper proves: "a model does not need to preserve every detail, only the consequences of actions that matter for value estimation."

**3. No policy prior for candidate generation**

We sampled 256 uniform random 5-step sequences from a space of 144^5 ≈ 6.1 billion possibilities. Coverage is 0.000004%. TD-MPC2 generates most candidates from the trained policy + small noise. This focuses search on plausible-good regions instead of pure randomness.

**4. CEM for discrete actions**

CEM was designed for continuous optimization. For discrete action spaces, each iteration reduces to "count which actions appeared in the top-k, then sample proportionally." The 0.8/0.2 smoothing prevents collapse but also prevents convergence. MPPI's soft exponential weighting handles noisy returns better than CEM's hard elite selection.

**5. Independent action sampling (no temporal correlation)**

Our CEM samples each timestep independently. The iCEM paper showed that using temporally correlated (colored) noise gives 2.7-22x sample efficiency and 1.2-10x performance gain. For FPS where consecutive actions need coherence (tracking a target, advancing on a position), independent sampling generates incoherent motion.

### What We Got Right

- The dynamics model itself: accurate state predictions, proper ensemble, Gaussian NLL loss, bootstrap aggregation
- The MPC code mechanics: CEM implementation, planning loop, evaluation structure
- The experimental design: 4-way comparison isolating specific failure modes

The failure was architectural, not code quality.

## When MPC Works

| Factor | MPC-friendly | MPC-hostile |
|--------|-------------|-------------|
| **Reward density** | Dense (every step informative) | Sparse (kills are rare events) |
| **Action space** | Low-dimensional continuous | High-dimensional discrete/combinatorial |
| **Dynamics** | Smooth, predictable | Chaotic, opponent-dependent |
| **Time pressure** | None (can spend 100ms+) | Real-time (16ms budget) |
| **Horizon needed** | Short (physical control) | Long (strategy, positioning) |
| **Model quality needed** | Moderate (continuous is forgiving) | Very high (discrete is unforgiving) |

### Where MPC excels

- **Robotics / continuous control**: PETS achieves competitive SAC performance with 8x fewer samples on HalfCheetah. TD-MPC2 solves 80+ continuous tasks with one hyperparameter set.
- **Short-horizon physical tasks**: Drone stabilization, manipulation, locomotion. The dynamics are smooth and the reward is dense (stay upright, reach target).
- **Tasks with good value functions**: When you have a terminal Q(s) that captures long-horizon value, even a 3-5 step planning horizon works because the Q-function handles the rest.

### Where MPC fails

- **FPS games**: Sparse rewards, chaotic dynamics, opponent non-stationarity, need for coherent multi-second behavior. Everything is MPC-hostile.
- **Any domain where per-step reward differences are below model noise**: The fundamental signal-to-noise problem.
- **Large discrete action spaces without a strong policy prior**: Random search in exponentially large spaces.

## What About Pokemon?

**MPC is the wrong tool. MCTS is the right one.**

Pokemon battles are turn-based, discrete, and have large immediate consequences. A move does 30% HP — not a 0.001 reward signal. This is exactly where tree search shines.

**Foul Play bot** achieved 90%+ GXE in Gen 9 Random Battles using MCTS. Key insight: switching from expectiminimax (limited to ~5 turns depth) to MCTS (10+ turns on promising lines, 2-3 on unpromising) was the breakthrough.

Why MCTS works for Pokemon:
- **Large, immediate consequences**: Each move does visible damage. Rewards are dense and discriminative.
- **Manageable branching factor**: ~4 moves + 5 switches = ~9 actions per turn. MCTS handles this easily.
- **Time to think**: No real-time pressure. Thousands of iterations per decision.
- **Deterministic-ish**: Damage formulas are known, RNG is bounded. Tree search can reason about outcomes.

**For GhostLobby's Pokemon Gold work**: The right architecture is MuZero-style — learned dynamics model + MCTS + neural value/policy network. NOT MPC/CEM.

## What Would Make MPC Work for CsLite?

If we wanted to fix MPC (rather than moving to Dreamer/MCTS), the minimum viable fix:

1. **Train a Q-function** via SAC or TD-learning on real gameplay. Use it as terminal value in the planner. This is the single biggest improvement.
2. **Learn a latent space** (encoder only, no decoder). Plan in latent space where model capacity focuses on task-relevant features.
3. **Use policy prior**: Generate 80% of candidates from the trained policy + noise, 20% random.
4. **Switch to MPPI**: Soft exponential weighting handles noisy returns better than hard elite selection.
5. **Add temporal correlation**: Color the noise so consecutive actions are coherent.

This is essentially building TD-MPC2 from scratch. At that point, forking the actual TD-MPC2 repo is more practical.

## The Taxonomy for GhostLobby's Games

| Game Type | Right Tool | Why |
|-----------|-----------|-----|
| **CsLite (real-time FPS)** | **Dreamer** (train in imagination) | Reactive policy needed at tick rate. Planning intelligence baked into weights via imagined experience. |
| **Pokemon Gold (turn-based RPG)** | **MCTS** (tree search) | Large discrete consequences, manageable branching, time to think. MuZero-style: learned model + MCTS. |
| **Tactical Deathmatch (2D)** | **Dreamer or TD-MPC2** | Slower pace than FPS, more time for planning. TD-MPC2's latent planning could work here. |
| **Drone hover (continuous)** | **TD-MPC2** (latent MPC) | Continuous control, dense reward, smooth dynamics. MPC's home territory. |

## References

- TD-MPC2: Hansen et al. (ICLR 2024) — [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)
- SLOPE: [arXiv:2602.03201](https://arxiv.org/abs/2602.03201) — fixes flat reward landscape for MBRL
- iCEM: [github.com/martius-lab/iCEM](https://github.com/martius-lab/iCEM) — temporally correlated CEM
- PETS: [arXiv:1805.12114](https://arxiv.org/abs/1805.12114) — ensemble MPC fundamentals
- MuZero analysis: [arXiv:2306.00840](https://arxiv.org/abs/2306.00840) — why decoder-free latent models work
- Foul Play Pokemon: [arXiv:2603.15563](https://arxiv.org/abs/2603.15563) — MCTS Pokemon bot, 90%+ GXE
- BMPC: [arXiv:2503.18871](https://arxiv.org/abs/2503.18871) — bootstrapped MPC (ICLR 2025)
