#!/usr/bin/env python3
"""Benchmark MLP vs LSTM vs GRU PPO throughput on CS-Lite.

Runs a short training session with each architecture and reports
steps/sec so we can decide if recurrent PPO is fast enough before
committing to a Sample Factory migration.
"""

import argparse
import os
import sys
import time

import torch as th
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

try:
    from sb3_contrib import RecurrentPPO
    from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
except ImportError:
    RecurrentPPO = None
    MlpLstmPolicy = None

from training.utils import resolve_config, make_vec_env
from training.ppo_trainer import _import_gym_class


# ---------------------------------------------------------------------------
# GRU policy: swap nn.LSTM for nn.GRU inside MlpLstmPolicy
# ---------------------------------------------------------------------------

if MlpLstmPolicy is not None:

    class MlpGruPolicy(MlpLstmPolicy):
        """MlpLstmPolicy with nn.GRU instead of nn.LSTM.

        GRU has 2 gates (vs LSTM's 3), fewer params, and returns a single
        hidden tensor instead of (h, c). We wrap the GRU state in a tuple
        to stay compatible with RecurrentPPO's state management.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Replace LSTM modules with GRU (same input/output dims)
            self.lstm_actor = nn.GRU(
                self.lstm_actor.input_size,
                self.lstm_actor.hidden_size,
                num_layers=self.lstm_actor.num_layers,
                batch_first=self.lstm_actor.batch_first,
            )
            if self.lstm_critic is not None:
                self.lstm_critic = nn.GRU(
                    self.lstm_critic.input_size,
                    self.lstm_critic.hidden_size,
                    num_layers=self.lstm_critic.num_layers,
                    batch_first=self.lstm_critic.batch_first,
                )

        @staticmethod
        def _process_sequence(features, lstm_states, episode_starts, lstm):
            """Override to handle GRU's single-tensor state format.

            RecurrentPPO passes states as (h, c) tuples everywhere.
            GRU only uses h, so we unwrap on input and re-wrap on output.
            """
            if isinstance(lstm, nn.GRU):
                h = lstm_states[0]
                n_seq = h.shape[1]

                features_sequence = features.reshape(
                    (n_seq, -1, lstm.input_size)
                ).swapaxes(0, 1)
                episode_starts_seq = episode_starts.reshape(
                    (n_seq, -1)
                ).swapaxes(0, 1)

                if th.all(episode_starts_seq == 0.0):
                    output, h_new = lstm(features_sequence, h)
                    output = th.flatten(
                        output.transpose(0, 1), start_dim=0, end_dim=1
                    )
                    return output, (h_new, h_new)

                outputs = []
                for feat, ep_start in zip(
                    features_sequence, episode_starts_seq, strict=True
                ):
                    h = (1.0 - ep_start).view(1, n_seq, 1) * h
                    hidden, h = lstm(feat.unsqueeze(0), h)
                    outputs.append(hidden)

                output = th.flatten(
                    th.cat(outputs).transpose(0, 1), start_dim=0, end_dim=1
                )
                return output, (h, h)
            else:
                return MlpLstmPolicy._process_sequence(
                    features, lstm_states, episode_starts, lstm
                )

else:
    MlpGruPolicy = None


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

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


def run_bench(label, algo_class, policy, policy_kwargs, vec_env, timesteps):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    cb = ThroughputBenchCallback()

    model = algo_class(
        policy,
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
    p = argparse.ArgumentParser(
        description="MLP vs LSTM vs GRU throughput benchmark"
    )
    p.add_argument("--config", default="configs/cs_lite/1v1_tactical.json")
    p.add_argument("--scenario", default="cs_lite_dummy")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--timesteps", type=int, default=200_000,
                   help="Steps per benchmark (default 200K)")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--frame-skip", type=int, default=4)
    args = p.parse_args()

    config_path = resolve_config(args.config)
    gym_class = _import_gym_class(args.scenario)

    print(f"Config: {config_path}")
    print(f"Scenario: {args.scenario}")
    print(f"Envs: {args.n_envs}")
    print(f"Steps per bench: {args.timesteps:,}")
    print(f"Hidden size: {args.hidden_size}")

    vec_env = make_vec_env(
        gym_class,
        config_path=config_path,
        n_envs=args.n_envs,
        scenario=args.scenario,
        frame_skip=args.frame_skip,
        max_steps=2048,
    )

    results = {}

    # --- MLP baseline ---
    results["MLP"] = run_bench(
        "MLP (MlpPolicy)",
        PPO, "MlpPolicy", None,
        vec_env, args.timesteps,
    )

    if RecurrentPPO is None:
        print("\nRecurrentPPO not available (pip install sb3-contrib)")
        vec_env.close()
        return

    # --- LSTM ---
    results["LSTM"] = run_bench(
        f"LSTM (MlpLstmPolicy, hidden={args.hidden_size})",
        RecurrentPPO, "MlpLstmPolicy",
        {"lstm_hidden_size": args.hidden_size},
        vec_env, args.timesteps,
    )

    # --- GRU ---
    if MlpGruPolicy is not None:
        results["GRU"] = run_bench(
            f"GRU (MlpGruPolicy, hidden={args.hidden_size})",
            RecurrentPPO, MlpGruPolicy,
            {"lstm_hidden_size": args.hidden_size},
            vec_env, args.timesteps,
        )

    # --- Summary ---
    mlp_tp = results["MLP"][0]

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    for name, (tp, elapsed) in results.items():
        ratio = mlp_tp / tp if tp > 0 else float("inf")
        slowdown = f"({ratio:.1f}x slower)" if name != "MLP" else "(baseline)"
        print(f"  {name:<6} {tp:>8,.0f} steps/sec  {elapsed:>6.1f}s  {slowdown}")
    print(f"{'='*60}")

    vec_env.close()


if __name__ == "__main__":
    main()
