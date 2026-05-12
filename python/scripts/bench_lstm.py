#!/usr/bin/env python3
"""Benchmark MLP vs LSTM PPO throughput on CS-Lite.

Runs a short training session with each architecture and reports
steps/sec so we can decide if RecurrentPPO is fast enough before
committing to a Sample Factory migration.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from training.utils import resolve_config, make_vec_env
from training.ppo_trainer import _import_gym_class


class ThroughputBenchCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.t0 = None
        self.steps_at_start = 0

    def _on_training_start(self):
        self.t0 = time.time()
        self.steps_at_start = self.num_timesteps

    def _on_step(self):
        return True

    def get_throughput(self):
        if self.t0 is None:
            return 0.0
        elapsed = time.time() - self.t0
        steps = self.num_timesteps - self.steps_at_start
        return steps / elapsed if elapsed > 0 else 0.0


def run_bench(label, algo_class, policy_name, policy_kwargs, vec_env, timesteps):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    cb = ThroughputBenchCallback()

    model = algo_class(
        policy_name,
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        **({"policy_kwargs": policy_kwargs} if policy_kwargs else {}),
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb)
    elapsed = time.time() - t0

    throughput = cb.get_throughput()
    print(f"  Steps: {timesteps:,}")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Throughput: {throughput:,.0f} steps/sec")

    del model
    return throughput, elapsed


def main():
    p = argparse.ArgumentParser(description="MLP vs LSTM throughput benchmark")
    p.add_argument("--config", default="configs/cs_lite/1v1_tactical.json")
    p.add_argument("--scenario", default="cs_lite_dummy")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--timesteps", type=int, default=200_000,
                   help="Steps per benchmark (default 200K, enough to measure steady-state)")
    p.add_argument("--lstm-hidden-size", type=int, default=64)
    p.add_argument("--frame-skip", type=int, default=4)
    args = p.parse_args()

    config_path = resolve_config(args.config)
    gym_class = _import_gym_class(args.scenario)

    print(f"Config: {config_path}")
    print(f"Scenario: {args.scenario}")
    print(f"Envs: {args.n_envs}")
    print(f"Steps per bench: {args.timesteps:,}")
    print(f"LSTM hidden size: {args.lstm_hidden_size}")

    vec_env = make_vec_env(
        gym_class,
        config_path=config_path,
        n_envs=args.n_envs,
        scenario=args.scenario,
        frame_skip=args.frame_skip,
        max_steps=2048,
    )

    # --- MLP baseline ---
    mlp_throughput, mlp_time = run_bench(
        "MLP (MlpPolicy)",
        PPO, "MlpPolicy", None,
        vec_env, args.timesteps,
    )

    # --- LSTM 64 ---
    if RecurrentPPO is None:
        print("\nRecurrentPPO not available (pip install sb3-contrib)")
        vec_env.close()
        return

    lstm_throughput, lstm_time = run_bench(
        f"LSTM (MlpLstmPolicy, hidden={args.lstm_hidden_size})",
        RecurrentPPO, "MlpLstmPolicy",
        {"lstm_hidden_size": args.lstm_hidden_size},
        vec_env, args.timesteps,
    )

    # --- Summary ---
    ratio = mlp_throughput / lstm_throughput if lstm_throughput > 0 else float("inf")

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  MLP:   {mlp_throughput:>8,.0f} steps/sec  ({mlp_time:.1f}s)")
    print(f"  LSTM:  {lstm_throughput:>8,.0f} steps/sec  ({lstm_time:.1f}s)")
    print(f"  Ratio: MLP is {ratio:.1f}x faster than LSTM")
    print(f"{'='*60}")

    vec_env.close()


if __name__ == "__main__":
    main()
