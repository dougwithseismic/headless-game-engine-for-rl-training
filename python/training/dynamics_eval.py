"""Evaluation utilities for the dynamics ensemble world model.

Provides structured evaluation: 1-step accuracy, N-step divergence,
ensemble disagreement analysis, and formatted reports.

Typical usage::

    from training.dynamics_eval import evaluate_world_model, print_report

    results = evaluate_world_model(ensemble, replay_buffer, policy)
    print_report(results)
"""

from __future__ import annotations

import numpy as np

from training.replay_buffer import ReplayBuffer
from training.dynamics_model import DynamicsEnsemble


def evaluate_1step(
    ensemble: DynamicsEnsemble,
    buffer: ReplayBuffer,
    n_samples: int = 2000,
) -> dict:
    """Evaluate 1-step prediction accuracy.

    Returns
    -------
    dict with keys:
        state_mse, reward_mse, per_feature_mse, r_squared,
        mean_disagreement, pct_features_above_95_r2
    """
    rng = np.random.default_rng(0)
    n = min(n_samples, len(buffer))
    batch = buffer.sample(n, rng=rng)

    pred_delta, pred_reward, disagreement = ensemble.predict(
        batch["observations"], batch["actions"]
    )
    pred_next = batch["observations"] + pred_delta
    actual_next = batch["next_observations"]

    per_feature_mse = np.mean((pred_next - actual_next) ** 2, axis=0)
    state_mse = float(per_feature_mse.mean())
    reward_mse = float(np.mean((pred_reward - batch["rewards"]) ** 2))

    actual_delta = actual_next - batch["observations"]
    delta_var = np.var(actual_delta, axis=0)
    delta_var = np.maximum(delta_var, 1e-8)
    pred_delta_err = np.mean((pred_delta - actual_delta) ** 2, axis=0)
    r_squared = 1.0 - pred_delta_err / delta_var

    pct_good = float(np.mean(r_squared > 0.95) * 100)

    return {
        "state_mse": state_mse,
        "reward_mse": reward_mse,
        "per_feature_mse": per_feature_mse,
        "r_squared": r_squared,
        "mean_disagreement": float(np.mean(disagreement)),
        "pct_features_above_95_r2": pct_good,
    }


def evaluate_nstep(
    ensemble: DynamicsEnsemble,
    buffer: ReplayBuffer,
    policy,
    horizons: list[int] | None = None,
    n_rollouts: int = 200,
) -> dict:
    """Evaluate N-step prediction divergence.

    Uses the dynamics model to roll forward from real states and measures
    how quickly predictions drift.

    Returns
    -------
    dict with keys per horizon:
        step_mse[h], step_disagreement[h], compounding_ratio
    """
    if horizons is None:
        horizons = [1, 3, 5, 10, 15, 20]

    max_h = max(horizons)
    rng = np.random.default_rng(1)
    starts = buffer.sample_states(n_rollouts, rng=rng)

    rollout = ensemble.rollout(
        starts, policy, horizon=max_h, deterministic=True
    )

    batch_1step = buffer.sample(n_rollouts, rng=rng)
    pred_1, _, _ = ensemble.predict_next_obs(
        batch_1step["observations"], batch_1step["actions"]
    )
    real_1step_mse = float(np.mean((pred_1 - batch_1step["next_observations"]) ** 2))

    results = {"real_1step_mse": real_1step_mse, "horizons": {}}

    for h in horizons:
        if h > max_h:
            continue
        disagree = float(rollout["disagreements"][:, :h].mean())
        drift = float(np.mean(np.abs(rollout["observations"][:, h] - starts)))
        results["horizons"][h] = {
            "disagreement": disagree,
            "drift": drift,
        }

    h1 = results["horizons"].get(1, {}).get("drift", 1e-8)
    h5 = results["horizons"].get(5, {}).get("drift", 0)
    results["compounding_ratio_5"] = h5 / max(h1, 1e-8)

    return results


def evaluate_world_model(
    ensemble: DynamicsEnsemble,
    buffer: ReplayBuffer,
    policy=None,
    n_1step: int = 2000,
    n_rollouts: int = 200,
    horizons: list[int] | None = None,
) -> dict:
    """Full evaluation suite for the dynamics model.

    Returns combined results from 1-step and N-step evaluations.
    """
    results = {"one_step": evaluate_1step(ensemble, buffer, n_1step)}

    if policy is not None:
        results["nstep"] = evaluate_nstep(
            ensemble, buffer, policy,
            horizons=horizons, n_rollouts=n_rollouts,
        )

    results["buffer_size"] = len(buffer)
    results["n_models"] = ensemble.n_models

    return results


def print_report(results: dict) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("  DYNAMICS MODEL EVALUATION REPORT")
    print("=" * 60)

    one = results["one_step"]
    print(f"\n  Buffer size:  {results['buffer_size']:,}")
    print(f"  Ensemble:     {results['n_models']} members")

    print(f"\n  --- 1-Step Prediction ---")
    print(f"  State MSE:       {one['state_mse']:.6f}")
    print(f"  Reward MSE:      {one['reward_mse']:.6f}")
    print(f"  Mean R-squared:  {float(np.median(one['r_squared'])):.4f} (median)")
    print(f"  Features R2>0.95:{one['pct_features_above_95_r2']:.1f}%")
    print(f"  Disagreement:    {one['mean_disagreement']:.6f}")

    r2 = one["r_squared"]
    worst_idx = np.argsort(r2)[:5]
    print(f"\n  Worst 5 features (by R2):")
    for idx in worst_idx:
        print(f"    feature[{idx:3d}]: R2={r2[idx]:.4f}, MSE={one['per_feature_mse'][idx]:.6f}")

    if "nstep" in results:
        ns = results["nstep"]
        print(f"\n  --- N-Step Divergence ---")
        print(f"  Real 1-step MSE: {ns['real_1step_mse']:.6f}")
        for h, data in sorted(ns["horizons"].items()):
            print(
                f"  {h:2d}-step: drift={data['drift']:.4f}, "
                f"disagree={data['disagreement']:.6f}"
            )
        print(f"  Compounding ratio (5/1): {ns.get('compounding_ratio_5', 0):.2f}x")

    print("\n  --- Quality Assessment ---")
    r2_median = float(np.median(one["r_squared"]))
    if r2_median > 0.95:
        print("  [GOOD] Median R2 > 0.95 -- model predictions are accurate")
    elif r2_median > 0.80:
        print("  [OK]   Median R2 > 0.80 -- model is usable for short rollouts")
    else:
        print("  [POOR] Median R2 < 0.80 -- model needs more data or tuning")

    if "nstep" in results:
        cr = results["nstep"].get("compounding_ratio_5", 0)
        if cr < 3.0:
            print("  [GOOD] 5-step drift < 3x 1-step -- sublinear compounding")
        elif cr < 10.0:
            print("  [OK]   5-step drift < 10x 1-step -- moderate compounding")
        else:
            print("  [WARN] 5-step drift > 10x 1-step -- rapid error compounding")

    print("=" * 60 + "\n")


def log_to_tensorboard(results: dict, writer, global_step: int) -> None:
    """Log evaluation results to TensorBoard."""
    one = results["one_step"]
    writer.add_scalar("dynamics_eval/state_mse", one["state_mse"], global_step)
    writer.add_scalar("dynamics_eval/reward_mse", one["reward_mse"], global_step)
    writer.add_scalar(
        "dynamics_eval/median_r_squared",
        float(np.median(one["r_squared"])),
        global_step,
    )
    writer.add_scalar(
        "dynamics_eval/pct_features_r2_95",
        one["pct_features_above_95_r2"],
        global_step,
    )
    writer.add_scalar(
        "dynamics_eval/disagreement",
        one["mean_disagreement"],
        global_step,
    )

    if "nstep" in results:
        ns = results["nstep"]
        writer.add_scalar(
            "dynamics_eval/real_1step_mse", ns["real_1step_mse"], global_step
        )
        for h, data in ns["horizons"].items():
            writer.add_scalar(
                f"dynamics_eval/{h}step_drift", data["drift"], global_step
            )
            writer.add_scalar(
                f"dynamics_eval/{h}step_disagree",
                data["disagreement"],
                global_step,
            )
