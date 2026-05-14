# Implementation Audit: Dreamer + TD-MPC v2

Four parallel audits completed against reference implementations. Both our from-scratch implementations have critical bugs that prevent learning entirely. This is not a scale problem — these bugs would prevent learning at any scale.

## Dreamer: 8 CRITICAL Bugs

| # | Bug | Impact |
|---|-----|--------|
| 1 | **KL balancing is a no-op** — no stop-gradients, formula just multiplies by alpha + (1-alpha) = 1 | Latent space collapses, world model can't learn meaningful representations |
| 2 | **GRU missing LayerNorm and update_bias=-1** — uses vanilla nn.GRUCell | Training unstable, GRU forgets too aggressively |
| 3 | **Imagination gradient chain broken** — numpy round-trip kills all gradients | Actor can't learn from imagined trajectories (the entire point of Dreamer) |
| 4 | **Actor uses torch.multinomial** — non-differentiable sampling | Zero gradients to actor, even REINFORCE doesn't work properly |
| 5 | **No is_first episode boundary handling** — GRU state bleeds across episodes | Corrupted gradients in world model training |
| 6 | **Actor outputs integers not straight-through one-hots** | Breaks dynamics gradient mode |
| 7 | **Critic is scalar MSE not two-hot DiscDist** — missing core DreamerV3 innovation | Critic unstable, can't represent multi-modal value distributions |
| 8 | **Advantage mixes symlog(critic) with normed returns** — incompatible scales | Actor loss meaningless |

Plus 10 SIGNIFICANT issues (missing LayerNorm everywhere, single imagination start state, wrong free nats application, no discount weighting, etc.)

## TD-MPC v2: 3 CRITICAL Bugs

| # | Bug | Impact |
|---|-----|--------|
| 1 | **Policy gradient is zero** — torch.multinomial + numpy round-trip | Policy network never updates, runs random initial weights |
| 2 | **Missing two-hot distributional RL** — scalar MSE instead of 101-bin categorical | Q-function and reward prediction unstable |
| 3 | **Loss not normalized by horizon** — value loss 15x larger than intended | Loss balance completely wrong |

Plus 3 SIGNIFICANT (policy only uses first timestep, MPPI temperature wrong, target encoder shouldn't exist).

## Root Cause

Both implementations share the same fundamental bug: **discrete actions are handled via torch.multinomial which is non-differentiable, and one_hot_actions round-trips through CPU numpy, severing the computation graph.** This means:

- In Dreamer: the actor receives zero useful gradients from imagination
- In TD-MPC: the policy optimizer steps on zero gradients every iteration

## Recommendation: Fork SheepRL

SheepRL is the only framework that:
- Natively supports MultiDiscrete([12, 2, 2, 3]) without flattening
- Has all DreamerV3 details correct (KL balancing, LayerNorm GRU, two-hot critic, straight-through sampling)
- Supports distributed training via Lightning Fabric
- Requires only ~20 lines of wrapper code + 2 YAML configs to integrate

Integration steps:
1. Write Dict obs wrapper around CsLiteGym (~20 lines)
2. Register with gymnasium.register()
3. Create env YAML + experiment YAML
4. Run: `python sheeprl.py exp=ghostlobby_dreamer_v3`

Estimated time: 1-2 hours. Expected training: 2-6 hours on 3090 for 1M steps.
