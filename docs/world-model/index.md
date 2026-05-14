# World Model Research: Gun Cleric Mode

Agents that predict combat futures before acting, not just react to the current frame. The name comes from Equilibrium's gun clerics -- fighters who pre-compute the geometry of a gunfight and move through the solution. The goal is the same thing for RL agents in GhostLobby: learn an internal dynamics model of the game, use it to simulate ahead, and act on predicted outcomes.

## Reading Order

### Research

1. **[direction.md](direction.md)** -- The full research landscape. Covers what world models are mechanically, the architecture spectrum (Dyna through Dreamer through IRIS), competitive analysis ($2.5B+ funding in world model companies), and the strategic rationale. Includes implementation plan, research paper roadmap, and open-source library recommendations.

### Phase 1: Dyna Integration

2. **[demo-results.md](demo-results.md)** -- First end-to-end demo on real CsLite 1v1 data. 398K transitions, trained 5-model ensemble in 43s on M4 Max. Proved the dynamics are learnable: reward MSE 0.0003, combat features R2 > 0.95, compounding ratio 4.1x at 5-step horizon.

3. **[ab-test-results.md](ab-test-results.md)** -- First A/B test: PPO vs PPO+Dyna, 2M steps each. Key finding: **stability** not speed. Dyna maintained 40% less reward variance during self-play instability. Identified that the callback collected data but didn't inject synthetic transitions -- the stability came from the replay buffer alone.

4. **[3way-ab-test-results.md](3way-ab-test-results.md)** -- 3-way A/B test with synthetic transition injection. Compared baseline PPO, Dyna (collect-only), and Dyna (reward shaping + curiosity). **Dyna collect-only wins on every metric**: peak 124.0 vs 121.9, 2nd-half avg 115.0 vs 107.2, collapses 10% vs 30%. Reward shaping didn't help in its current form -- the replay buffer temporal smoothing is the mechanism.

### Phase 2: MPC + Dreamer

5. **[phase2-mpc-dreamer.md](phase2-mpc-dreamer.md)** -- Deep research into the two gun cleric architectures: MPC (plan ahead at each step) and Dreamer (train in imagination). Covers TD-MPC2, DreamerV3/V4, Dream-MPC hybrid, and how they map to the hierarchical agent framework.

6. **[mpc-experiment-results.md](mpc-experiment-results.md)** -- MPC experiment on real CsLite combat. **MPC failed catastrophically** — reward dropped from 118 to 4-6. Root cause: flat reward landscape + no terminal value function + observation-space planning.

7. **[mpc-when-it-works.md](mpc-when-it-works.md)** -- Post-mortem: what we got wrong (5 specific architectural gaps), when MPC actually works (robotics, continuous control, dense rewards), when to use MCTS instead (Pokemon, turn-based games), and the right tool for each GhostLobby game type.

8. **[tdmpc-experiment-results.md](tdmpc-experiment-results.md)** -- Proper MPC with all 5 fixes (latent space, Q-function, MPPI, policy prior). The learned policy head matches PPO (117.6 vs 118.8) but planning still fails (9.9) due to offline Q-function overestimation.

9. **[online-training-results.md](online-training-results.md)** -- Online Dreamer + TD-MPC v2 with real env interaction. Both architecturally correct but flat at ~3.0 reward due to insufficient model capacity and step budget.

10. **[status-and-next.md](status-and-next.md)** -- Status at end of from-scratch phase.

11. **[implementation-audit.md](implementation-audit.md)** -- Full audit results: 8 CRITICAL bugs in Dreamer, 3 in TD-MPC v2. Root cause: non-differentiable action sampling + KL balancing no-op. Both implementations fundamentally broken, not just underscaled. **Recommendation: fork SheepRL** (natively supports MultiDiscrete, 20 lines of wrapper code).

### Reference

7. **[next-steps.md](next-steps.md)** -- Original Phase 1 next-steps. Partially superseded.

## Current Status

**Phase 1 (Dyna) and Phase 2 (MPC) both complete. The dynamics model works; neither integration approach achieved gun cleric mode.**

**What worked:**
- Game dynamics are learnable from state vectors in 43s on M4 Max (R2 > 0.95 on combat features)
- Expert data produces 35x better R2 than random policy data
- Replay buffer stabilizes self-play: collapse rate 30% → 10%, peak reward 120.4 → 124.0

**What didn't:**
- Reward shaping on PPO: PPO is reactive, non-stationary reward signals add noise
- MPC planning at observation level: per-step reward landscape is too flat, planner optimizes noise and destroys coherent behavior (reward 118 → 4)
- GATED MPC: gate barely opens (2.6%) because model can't distinguish good from bad actions

**Why MPC failed but Dreamer should work:**
MPC asks "model, what should I do?" and the model says "all actions look the same." Dreamer asks "model, give me imagined experience" and the actor-critic learns from it like real experience. The model doesn't need to rank actions — it just needs to produce realistic trajectories. It does.

**The path to gun cleric:** Either (a) Dreamer — train the policy in imagination so "thinking ahead" is baked into the weights, or (b) latent-space MPC (TD-MPC2 style) where planning happens in a learned representation with structured reward landscape, not raw observation space.

## Code

```
python/training/replay_buffer.py      # FIFO replay buffer
python/training/dynamics_model.py     # MLP ensemble (predict next state + reward)
python/training/dynamics_trainer.py   # Training orchestrator
python/training/dyna_callback.py      # SB3 callback (collect + train + shape)
python/training/dynamics_eval.py      # Evaluation suite
python/training/mpc_planner.py        # MPC planner (random shooting + CEM)
python/scripts/demo_world_model.py    # End-to-end demo (--policy for expert data)
python/scripts/analyze_ab_test.py     # A/B test analysis
python/scripts/eval_mpc.py            # MPC evaluation (4-way comparison)
```

Usage:
```bash
# Demo with expert data
python scripts/demo_world_model.py --config configs/cs_lite/1v1_wide.json \
  --policy runs/<best_model>/best/best_model.zip --episodes 300

# Train with Dyna (recommended: no-shaping for self-play)
python scripts/train.py --scenario cs_lite --config configs/cs_lite/1v1_wide.json \
  --dyna --dyna-no-shaping --self-play --timesteps 5000000
```
