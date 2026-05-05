"""
Throughput benchmark for GhostLobby environments.

Measures raw env stepping speed (no training) across configurations
to find the ceiling before PyTorch/PPO overhead enters the picture.

Usage:
    python python/benchmark_throughput.py
    python python/benchmark_throughput.py --config configs/oval_race.json --scenario racing
    python python/benchmark_throughput.py --envs 1,2,4,8,16 --duration 10
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_config(path):
    return os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path


def bench_single_env(config_path, scenario, frame_skip, duration):
    from ghostlobby_gym import GhostLobbyGym

    env = GhostLobbyGym(config_path=config_path, scenario=scenario, frame_skip=frame_skip, max_steps=999999)
    obs, _ = env.reset()
    action = env.action_space.sample()

    # warmup
    for _ in range(100):
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()

    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
        steps += 1

    elapsed = time.perf_counter() - t0
    return steps, elapsed


def bench_subproc_vecenv(config_path, scenario, frame_skip, n_envs, duration):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from ghostlobby_gym import GhostLobbyGym

    def make():
        def _init():
            return GhostLobbyGym(config_path=config_path, scenario=scenario, frame_skip=frame_skip, max_steps=999999)
        return _init

    vec_env = SubprocVecEnv([make() for _ in range(n_envs)])
    vec_env.reset()
    actions = np.stack([vec_env.action_space.sample() for _ in range(n_envs)])

    # warmup
    for _ in range(50):
        vec_env.step(actions)

    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        vec_env.step(actions)
        steps += n_envs

    elapsed = time.perf_counter() - t0
    vec_env.close()
    return steps, elapsed


def bench_with_inference(config_path, scenario, frame_skip, n_envs, duration, obs_size):
    """SubprocVecEnv + simulated MLP forward pass to approximate real training."""
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from ghostlobby_gym import GhostLobbyGym

    def make():
        def _init():
            return GhostLobbyGym(config_path=config_path, scenario=scenario, frame_skip=frame_skip, max_steps=999999)
        return _init

    vec_env = SubprocVecEnv([make() for _ in range(n_envs)])
    obs = vec_env.reset()

    policy = torch.nn.Sequential(
        torch.nn.Linear(obs_size, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, vec_env.action_space.shape[0]),
    )
    policy.eval()

    # warmup
    for _ in range(50):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            actions = policy(obs_t).numpy()
        obs, _, dones, _ = vec_env.step(actions)

    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            actions = policy(obs_t).numpy()
        obs, _, dones, _ = vec_env.step(actions)
        steps += n_envs

    elapsed = time.perf_counter() - t0
    vec_env.close()
    return steps, elapsed


def format_rate(steps, elapsed):
    sps = steps / elapsed
    if sps >= 1000:
        return f"{sps:,.0f}"
    return f"{sps:,.1f}"


def main():
    p = argparse.ArgumentParser(description="GhostLobby throughput benchmark")
    p.add_argument("--config", default="configs/1v1_deathmatch.json")
    p.add_argument("--scenario", default="fps")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--duration", type=float, default=5.0, help="Seconds per test")
    p.add_argument("--envs", default="1,2,4,8", help="Comma-separated env counts to test")
    args = p.parse_args()

    config = resolve_config(args.config)
    env_counts = [int(x) for x in args.envs.split(",")]

    print(f"Config:     {args.config}")
    print(f"Scenario:   {args.scenario}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"Duration:   {args.duration}s per test")
    print()

    # Phase 1: raw single env
    print("=" * 65)
    print("Phase 1: Raw env stepping (no policy, no vectorization)")
    print("=" * 65)
    steps, elapsed = bench_single_env(config, args.scenario, args.frame_skip, args.duration)
    sps = steps / elapsed
    tps = sps * args.frame_skip
    obs_size = None

    from ghostlobby_gym import GhostLobbyGym
    probe = GhostLobbyGym(config_path=config, scenario=args.scenario, frame_skip=args.frame_skip, max_steps=100)
    obs_size = probe.observation_space.shape[0]
    del probe

    print(f"  Steps/sec:  {format_rate(steps, elapsed):>10}")
    print(f"  Ticks/sec:  {format_rate(int(tps * elapsed), elapsed):>10}  (steps x frame_skip)")
    print()

    # Phase 2: SubprocVecEnv scaling
    print("=" * 65)
    print("Phase 2: SubprocVecEnv scaling (random actions, no policy)")
    print("=" * 65)
    print(f"  {'Envs':>6}  {'Steps/sec':>12}  {'Ticks/sec':>12}  {'Scaling':>8}")
    print(f"  {'----':>6}  {'--------':>12}  {'--------':>12}  {'-------':>8}")

    baseline_sps = None
    for n in env_counts:
        if n == 1:
            s, e = bench_single_env(config, args.scenario, args.frame_skip, args.duration)
        else:
            s, e = bench_subproc_vecenv(config, args.scenario, args.frame_skip, n, args.duration)
        sps = s / e
        if baseline_sps is None:
            baseline_sps = sps
        scaling = sps / baseline_sps
        print(f"  {n:>6}  {format_rate(s, e):>12}  {format_rate(int(sps * args.frame_skip * e), e):>12}  {scaling:>7.1f}x")
    print()

    # Phase 3: with simulated inference
    print("=" * 65)
    print("Phase 3: SubprocVecEnv + MLP forward pass (simulated training)")
    print("=" * 65)
    print(f"  {'Envs':>6}  {'Steps/sec':>12}  {'Ticks/sec':>12}  {'vs raw':>8}")
    print(f"  {'----':>6}  {'--------':>12}  {'--------':>12}  {'------':>8}")

    for n in env_counts:
        s, e = bench_with_inference(config, args.scenario, args.frame_skip, n, args.duration, obs_size)
        sps = s / e
        raw_s, raw_e = (bench_single_env if n == 1 else lambda *a: bench_subproc_vecenv(*a))(
            config, args.scenario, args.frame_skip, *([n, args.duration] if n > 1 else [args.duration])
        )
        raw_sps = raw_s / raw_e
        overhead = sps / raw_sps
        print(f"  {n:>6}  {format_rate(s, e):>12}  {format_rate(int(sps * args.frame_skip * e), e):>12}  {overhead:>7.0%}")
    print()

    print("=" * 65)
    print("Interpretation:")
    print("  Phase 1 = engine ceiling (what Rust can deliver)")
    print("  Phase 2 = IPC overhead (subprocess pickling cost)")
    print("  Phase 3 = inference overhead (Phase 2 + policy forward pass)")
    print("  Gap between Phase 2 and 3 = time spent in PyTorch")
    print("  Gap between Phase 3 and actual PPO = gradient updates + GAE + buffer management")
    print("=" * 65)


if __name__ == "__main__":
    main()
