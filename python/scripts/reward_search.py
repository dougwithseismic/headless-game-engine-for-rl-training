#!/usr/bin/env python3
"""Batch reward search — run multiple reward configs and compare behavioral metrics."""

import argparse
import json
import os
import sys
import tempfile
import copy
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def merge_reward_config(base_config_path: str, reward_overrides: dict, output_path: str) -> str:
    """Merge reward overrides into a base config and write to output_path."""
    with open(base_config_path) as f:
        config = json.load(f)

    extra = config.setdefault("extra", {})
    for key, value in reward_overrides.items():
        extra[key] = value

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    return output_path


def run_candidate(scenario, config_path, bc_model, steps, n_envs, eval_freq, run_name):
    """Run a single training candidate and return behavioral metrics."""
    from training.ppo_trainer import PPOTrainer

    trainer = PPOTrainer(
        scenario=scenario,
        config_path=config_path,
        name=run_name,
        timesteps=steps,
        n_envs=n_envs,
        n_eval_episodes=16,
        eval_freq=eval_freq,
        checkpoint_freq=steps + 1,  # no checkpoints during search
        resume=bc_model,
        auto_stop=False,
        track_behavior=True,
        n_steps=4096,
        batch_size=256,
    )

    final_model = trainer.train()

    # Read metrics from eval logs
    eval_path = os.path.join(trainer.run_dir, "eval_logs", "evaluations.npz")
    metrics = {"reward": 0.0, "accuracy": 0.0, "kills_per_ep": 0.0,
               "deaths_per_ep": 0.0, "shoot_rate": 0.0, "damage_dealt_per_ep": 0.0}

    if os.path.exists(eval_path):
        data = np.load(eval_path)
        if len(data["results"]) > 0:
            metrics["reward"] = float(data["results"][-1].mean())

    # Read from TensorBoard
    try:
        import glob
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tb_files = glob.glob(os.path.join(trainer.run_dir, "tb", "**", "events*"), recursive=True)
        if tb_files:
            ea = EventAccumulator(tb_files[0])
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            tag_map = {
                "behavior/accuracy": "accuracy",
                "behavior/kills_per_ep": "kills_per_ep",
                "behavior/deaths_per_ep": "deaths_per_ep",
                "behavior/shoot_rate": "shoot_rate",
                "behavior/damage_dealt_per_ep": "damage_dealt_per_ep",
            }
            for tb_tag, key in tag_map.items():
                if tb_tag in tags:
                    vals = ea.Scalars(tb_tag)
                    if vals:
                        metrics[key] = float(vals[-1].value)
    except Exception:
        pass

    return metrics


def main():
    p = argparse.ArgumentParser(description="Batch reward search")
    p.add_argument("--base-config", required=True, help="Base game config JSON")
    p.add_argument("--reward-dir", required=True, help="Directory of reward override JSONs")
    p.add_argument("--scenario", default="cs_lite_dummy", help="Scenario name")
    p.add_argument("--steps", type=int, default=500_000, help="Steps per candidate")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--bc-model", default=None, help="BC model to warm-start from")
    p.add_argument("--output", default="data/reward_search_results.json")
    args = p.parse_args()

    # Discover reward configs
    reward_files = sorted([
        f for f in os.listdir(args.reward_dir)
        if f.endswith(".json")
    ])

    if not reward_files:
        print(f"No .json files found in {args.reward_dir}")
        sys.exit(1)

    print(f"=== Reward Search: {len(reward_files)} candidates, {args.steps:,} steps each ===")
    print()

    results = []

    for i, reward_file in enumerate(reward_files):
        reward_path = os.path.join(args.reward_dir, reward_file)
        with open(reward_path) as f:
            overrides = json.load(f)

        candidate_name = reward_file.replace(".json", "")
        print(f"[{i+1}/{len(reward_files)}] Running {candidate_name}...")
        print(f"  Overrides: {overrides}")

        # Merge into temp config
        tmp_config = os.path.join(tempfile.gettempdir(), f"reward_search_{candidate_name}.json")
        merge_reward_config(args.base_config, overrides, tmp_config)

        t0 = time.time()
        metrics = run_candidate(
            scenario=args.scenario,
            config_path=tmp_config,
            bc_model=args.bc_model,
            steps=args.steps,
            n_envs=args.n_envs,
            eval_freq=args.eval_freq,
            run_name=f"search_{candidate_name}",
        )
        elapsed = time.time() - t0

        metrics["name"] = candidate_name
        metrics["elapsed_s"] = elapsed
        metrics["overrides"] = overrides
        results.append(metrics)

        print(f"  Done in {elapsed:.0f}s: accuracy={metrics['accuracy']:.2%} "
              f"kills={metrics['kills_per_ep']:.1f} reward={metrics['reward']:.1f}")
        print()

    # Print comparison table
    print("=" * 90)
    print(f"{'Candidate':<28} {'Accuracy':>10} {'Kills/ep':>10} {'Deaths/ep':>10} "
          f"{'Shoot%':>8} {'Reward':>10}")
    print("-" * 90)

    best_acc = max(results, key=lambda r: r["accuracy"])
    best_kills = max(results, key=lambda r: r["kills_per_ep"])
    best_reward = max(results, key=lambda r: r["reward"])

    for r in sorted(results, key=lambda r: -r["accuracy"]):
        markers = []
        if r is best_acc: markers.append("best acc")
        if r is best_kills: markers.append("best kills")
        if r is best_reward: markers.append("best reward")
        marker = f"  <- {', '.join(markers)}" if markers else ""

        print(f"{r['name']:<28} {r['accuracy']:>9.2%} {r['kills_per_ep']:>10.1f} "
              f"{r['deaths_per_ep']:>10.1f} {r['shoot_rate']:>7.1%} "
              f"{r['reward']:>+10.1f}{marker}")

    print("=" * 90)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
