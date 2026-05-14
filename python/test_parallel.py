"""Test parallel emulators — the key to fast training."""
import numpy as np
import time
import os

from bridges.emulators.pyboy_host import PyBoyHost
from bridges.sinks.pyboy_sink import PyBoyActionSink
from bridges.sources.pyboy_source import PyBoyObservationSource, RAMFeature
from bridges.resets.pyboy_reset import PyBoyReset
from bridges.core.bridge import GameBridge, GameBridgeConfig
from bridges.core.timing import TimingConfig, TimingPolicy
from glgym.gym_external import ExternalGameGym

ROM = "data/pokemon_gold.gbc"
STATE = "data/pokemon_gold_ready.state"


def make_env():
    host = PyBoyHost(ROM, headless=True, speed=0, cgb=True)
    host.start()
    host.load_state_from_file(STATE)
    reset = PyBoyReset(host, auto_save_initial=True)
    bridge = GameBridge(
        action_sink=PyBoyActionSink(host),
        observation_source=PyBoyObservationSource(
            host,
            ram_features=[RAMFeature("c0", 0xC000, 255.0), RAMFeature("c1", 0xC001, 255.0)],
            ticks_per_step=8, render=False,
        ),
        reset_strategy=reset,
        config=GameBridgeConfig(name="gold", timing=TimingConfig(policy=TimingPolicy.FREE_RUNNING)),
    )
    return ExternalGameGym(bridge=bridge, max_steps=100)


if __name__ == "__main__":
    # Single env baseline
    print("=== Single env throughput ===")
    env = make_env()
    obs, _ = env.reset()
    t0 = time.monotonic()
    steps = 0
    for _ in range(500):
        obs, r, t, tr, info = env.step(env.action_space.sample())
        steps += 1
        if t or tr:
            obs, _ = env.reset()
    elapsed = time.monotonic() - t0
    single_sps = steps / elapsed
    print(f"  {steps} steps in {elapsed:.2f}s = {single_sps:.0f} steps/sec")
    env.close()

    # DummyVecEnv (same process, sanity check)
    print("\n=== DummyVecEnv (same process) ===")
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    for n_envs in [2, 4]:
        vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
        obs = vec_env.reset()
        t0 = time.monotonic()
        total_steps = 0
        for _ in range(200):
            actions = np.array([vec_env.action_space.sample() for _ in range(n_envs)])
            obs, rewards, dones, infos = vec_env.step(actions)
            total_steps += n_envs
        elapsed = time.monotonic() - t0
        sps = total_steps / elapsed
        print(f"  DummyVecEnv x{n_envs}: {total_steps} steps in {elapsed:.2f}s = {sps:.0f} steps/sec ({sps/single_sps:.1f}x)")
        vec_env.close()

    # SubprocVecEnv (true parallelism)
    print("\n=== SubprocVecEnv (parallel processes) ===")
    for n_envs in [2, 4, 8]:
        try:
            vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
            obs = vec_env.reset()
            t0 = time.monotonic()
            total_steps = 0
            for _ in range(200):
                actions = np.array([vec_env.action_space.sample() for _ in range(n_envs)])
                obs, rewards, dones, infos = vec_env.step(actions)
                total_steps += n_envs
            elapsed = time.monotonic() - t0
            sps = total_steps / elapsed
            print(f"  SubprocVecEnv x{n_envs}: {total_steps} steps in {elapsed:.2f}s = {sps:.0f} steps/sec ({sps/single_sps:.1f}x)")
            vec_env.close()
        except Exception as e:
            print(f"  SubprocVecEnv x{n_envs}: FAILED — {e}")

    print("\nDone.")
