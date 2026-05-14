#!/usr/bin/env python3
"""Analyze A/B test results: baseline PPO vs PPO+Dyna.

Usage:
    python scripts/analyze_ab_test.py --baseline runs/ab_baseline_* --dyna runs/ab_dyna_*
    python scripts/analyze_ab_test.py  # auto-discovers latest ab_ runs
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def load_eval_data(run_dir):
    """Load evaluation results from a run directory."""
    eval_path = os.path.join(run_dir, "eval_logs", "evaluations.npz")
    if not os.path.exists(eval_path):
        return None
    data = np.load(eval_path)
    timesteps = data["timesteps"]
    results = data["results"]
    if results.ndim > 1:
        mean_rewards = np.mean(results, axis=1)
    else:
        mean_rewards = results
    return {"timesteps": timesteps, "mean_rewards": mean_rewards, "raw": results}


def find_threshold_step(timesteps, rewards, threshold):
    """Find first timestep where reward exceeds threshold."""
    mask = rewards >= threshold
    if not mask.any():
        return None
    return int(timesteps[np.argmax(mask)])


def analyze(baseline_dir, dyna_dir):
    """Run full analysis comparing baseline and dyna runs."""
    base = load_eval_data(baseline_dir)
    dyna = load_eval_data(dyna_dir)

    if base is None or dyna is None:
        print("ERROR: Could not load eval data from one or both runs")
        return

    print("\n" + "=" * 70)
    print("  A/B TEST RESULTS: PPO (baseline) vs PPO+Dyna (world model)")
    print("=" * 70)

    print(f"\n  Baseline: {os.path.basename(baseline_dir)}")
    print(f"  Dyna:     {os.path.basename(dyna_dir)}")

    print(f"\n  --- Eval Progression ---")
    print(f"  {'Step':>10}  {'Baseline':>10}  {'Dyna':>10}  {'Delta':>10}  {'Winner':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    all_steps = sorted(set(base["timesteps"].tolist() + dyna["timesteps"].tolist()))
    base_interp = np.interp(all_steps, base["timesteps"], base["mean_rewards"])
    dyna_interp = np.interp(all_steps, dyna["timesteps"], dyna["mean_rewards"])

    for step in all_steps:
        if step % 100000 != 0 and step != all_steps[-1]:
            continue
        idx = all_steps.index(step)
        b = base_interp[idx]
        d = dyna_interp[idx]
        delta = d - b
        winner = "DYNA" if delta > 1 else ("BASE" if delta < -1 else "TIE")
        print(f"  {step:>10,}  {b:>10.1f}  {d:>10.1f}  {delta:>+10.1f}  {winner:>10}")

    print(f"\n  --- Summary Statistics ---")
    print(f"  Baseline: {len(base['timesteps'])} evals, "
          f"best={float(base['mean_rewards'].max()):.1f}, "
          f"last={float(base['mean_rewards'][-1]):.1f}, "
          f"final_step={int(base['timesteps'][-1]):,}")
    print(f"  Dyna:     {len(dyna['timesteps'])} evals, "
          f"best={float(dyna['mean_rewards'].max()):.1f}, "
          f"last={float(dyna['mean_rewards'][-1]):.1f}, "
          f"final_step={int(dyna['timesteps'][-1]):,}")

    thresholds = [100, 120, 150, 175, 200, 225, 250]
    print(f"\n  --- Steps to Reach Reward Threshold ---")
    print(f"  {'Threshold':>10}  {'Baseline':>12}  {'Dyna':>12}  {'Speedup':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}")

    for t in thresholds:
        b_step = find_threshold_step(base["timesteps"], base["mean_rewards"], t)
        d_step = find_threshold_step(dyna["timesteps"], dyna["mean_rewards"], t)
        b_str = f"{b_step:>12,}" if b_step else "     never"
        d_str = f"{d_step:>12,}" if d_step else "     never"
        if b_step and d_step:
            speedup = f"{b_step / d_step:.2f}x"
        elif d_step and not b_step:
            speedup = "DYNA only"
        elif b_step and not d_step:
            speedup = "BASE only"
        else:
            speedup = "neither"
        print(f"  {t:>10}  {b_str:>12}  {d_str:>12}  {speedup:>10}")

    b_area = float(np.trapezoid(base["mean_rewards"], base["timesteps"]))
    d_area = float(np.trapezoid(dyna["mean_rewards"], dyna["timesteps"]))
    print(f"\n  --- Area Under Reward Curve ---")
    print(f"  Baseline: {b_area:,.0f}")
    print(f"  Dyna:     {d_area:,.0f}")
    if b_area > 0:
        print(f"  Ratio:    {d_area / b_area:.3f}x")

    print(f"\n  --- Verdict ---")
    b_best = float(base["mean_rewards"].max())
    d_best = float(dyna["mean_rewards"].max())
    b_last = float(base["mean_rewards"][-1])
    d_last = float(dyna["mean_rewards"][-1])

    if d_best > b_best * 1.05:
        print(f"  Dyna reached a HIGHER peak reward ({d_best:.1f} vs {b_best:.1f})")
    elif b_best > d_best * 1.05:
        print(f"  Baseline reached a higher peak reward ({b_best:.1f} vs {d_best:.1f})")
    else:
        print(f"  Peak rewards are similar ({b_best:.1f} vs {d_best:.1f})")

    if d_area > b_area * 1.1:
        print(f"  Dyna LEARNED FASTER (higher area under curve)")
    elif b_area > d_area * 1.1:
        print(f"  Baseline learned faster (higher area under curve)")
    else:
        print(f"  Learning speed is similar")

    print("=" * 70 + "\n")


def main():
    p = argparse.ArgumentParser(description="Analyze A/B test")
    p.add_argument("--baseline", default=None)
    p.add_argument("--dyna", default=None)
    args = p.parse_args()

    if args.baseline and args.dyna:
        baseline_dir = glob.glob(args.baseline)[0] if "*" in args.baseline else args.baseline
        dyna_dir = glob.glob(args.dyna)[0] if "*" in args.dyna else args.dyna
    else:
        base_dirs = sorted(glob.glob("runs/ab_baseline_*"))
        dyna_dirs = sorted(glob.glob("runs/ab_dyna_*"))
        if not base_dirs or not dyna_dirs:
            print("No A/B test runs found. Use --baseline and --dyna flags.")
            sys.exit(1)
        baseline_dir = base_dirs[-1]
        dyna_dir = dyna_dirs[-1]

    analyze(baseline_dir, dyna_dir)


if __name__ == "__main__":
    main()
