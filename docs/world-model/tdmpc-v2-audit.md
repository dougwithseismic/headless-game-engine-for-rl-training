# TD-MPC v2 Audit Results

## Verification of 8 Claimed Fixes

| Fix | Status |
|-----|--------|
| 1. Multi-step training | Partially correct (missing loss normalization by H) |
| 2. SimNorm | Correct |
| 3. 20x consistency weight | Correct (with caveat from #1) |
| 4. Q-maximization policy | Correct intent, but BROKEN gradient flow (see Critical #3) |
| 5. Separate optimizers | Correct |
| 6. Correct consistency target | Correct (online encoder, not target) |
| 7. Proper MPPI | Partially correct (missing mean/std tracking, wrong temperature) |
| 8. Rho weighting | Correct |

## 3 CRITICAL Issues

### 1. Policy gradient is ZERO (non-differentiable sampling)

`_policy_sample` uses `torch.multinomial` (non-differentiable) then `one_hot_actions` round-trips through CPU numpy. The Q-value computation has NO gradient path back to policy parameters. The policy optimizer steps on zero gradients — **the policy network never updates**.

**Fix:** Use Gumbel-Softmax (straight-through) for differentiable discrete sampling.

### 2. Missing two-hot distributional RL

Reference uses 101-bin categorical distribution for rewards and Q-values with soft cross-entropy loss. We use scalar MSE. This is a fundamental stability mechanism.

### 3. No loss normalization by horizon

Losses are summed over H steps but never divided by H. With H=3 and 5 Q-heads, value loss is 15x larger than intended relative to consistency loss.

## 3 SIGNIFICANT Issues

4. Policy loss only uses first timestep (should use full zs sequence with rho weighting)
5. MPPI: hardcoded temperature 10.0 (should be 0.5), no mean/std tracking, no warm-starting
7. Target encoder shouldn't exist (reference only targets Q-heads, not encoder)

## 3 MINOR Issues

6. Missing encoder LR scaling (0.3x)
9. Q-scaling centers instead of just dividing
10. one_hot_actions CPU round-trip

## Root Cause of Failure

**Issue #1 (zero policy gradients) explains everything.** The policy network weights are static after initialization — it's running random actions from its initial weights. No amount of training steps or model capacity fixes zero gradients.
