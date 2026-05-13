#!/usr/bin/env python3
"""Benchmark MLP vs LSTM vs GRU training throughput.

Uses a minimal Box env to isolate the training loop overhead
from environment simulation cost. Reports steps/sec for each.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO


class FastBoxEnv(gym.Env):
    """Minimal env — instant steps, no compute."""

    def __init__(self, obs_dim=64, act_dim=8):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(act_dim,), dtype=np.float32)
        self._step = 0

    def reset(self, seed=None, options=None):
        self._step = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.random.randn(*self.observation_space.shape).astype(np.float32) * 0.1
        reward = float(np.random.randn())
        done = self._step >= 256
        return obs, reward, done, False, {}


def make_env():
    return FastBoxEnv()


def bench(label, algo_cls, policy, n_envs=8, total_steps=50_000, policy_kwargs=None):
    env = DummyVecEnv([make_env for _ in range(n_envs)])

    kwargs = dict(
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        verbose=0,
    )
    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs

    model = algo_cls(policy, env, **kwargs)

    # Warmup
    model.learn(total_timesteps=2048)

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_steps)
    elapsed = time.perf_counter() - t0

    sps = total_steps / elapsed
    print(f"  {label:>8s}: {sps:>8,.0f} steps/sec  ({elapsed:.1f}s for {total_steps:,} steps)")
    env.close()
    return sps


def main():
    total = 100_000
    n_envs = 8
    hidden = 256

    print(f"\n=== Memory Architecture Benchmark ===")
    print(f"  Env: FastBoxEnv (obs=64, act=8)")
    print(f"  Steps: {total:,}, Envs: {n_envs}, Hidden: {hidden}")
    print(f"  n_steps=512, batch=256, epochs=4")
    print()

    # MLP
    mlp_sps = bench("MLP", PPO, "MlpPolicy", n_envs=n_envs, total_steps=total)

    # LSTM
    lstm_sps = bench(
        "LSTM", RecurrentPPO, "MlpLstmPolicy",
        n_envs=n_envs, total_steps=total,
        policy_kwargs={"lstm_hidden_size": hidden},
    )

    # GRU
    from training.gru_policy import MlpGruPolicy
    gru_sps = bench(
        "GRU", RecurrentPPO, MlpGruPolicy,
        n_envs=n_envs, total_steps=total,
        policy_kwargs={"lstm_hidden_size": hidden},
    )

    print()
    print(f"  LSTM vs MLP: {mlp_sps / lstm_sps:.1f}x slower")
    print(f"   GRU vs MLP: {mlp_sps / gru_sps:.1f}x slower")
    print(f"  GRU vs LSTM: {gru_sps / lstm_sps:.2f}x")
    print()


if __name__ == "__main__":
    main()
