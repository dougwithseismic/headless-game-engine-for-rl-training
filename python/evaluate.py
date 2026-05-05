import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ghostlobby_gym import GhostLobbyGym


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained GhostLobby agent")
    p.add_argument("--model", required=True, help="Path to model .zip file")
    p.add_argument("--norm", default=None, help="Path to vec_normalize.pkl (auto-detected if not set)")
    p.add_argument("--config", default="configs/1v1_deathmatch.json", help="Game config")
    p.add_argument("--scenario", default="fps", help="Scenario name")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    p.add_argument("--frame-skip", type=int, default=4, help="Ticks per step")
    p.add_argument("--max-steps", type=int, default=2048, help="Max steps per episode")
    return p.parse_args()


def find_norm_file(model_path):
    run_dir = os.path.dirname(model_path)
    for candidate in [
        os.path.join(run_dir, "vec_normalize.pkl"),
        os.path.join(os.path.dirname(run_dir), "vec_normalize.pkl"),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    args = parse_args()

    norm_path = args.norm or find_norm_file(args.model)

    def make_env():
        return GhostLobbyGym(
            config_path=args.config,
            scenario=args.scenario,
            frame_skip=args.frame_skip,
            max_steps=args.max_steps,
        )

    if norm_path and os.path.exists(norm_path):
        print(f"Loading normalization from {norm_path}")
        env = VecNormalize.load(norm_path, DummyVecEnv([make_env]))
        env.training = False
        env.norm_reward = False
        use_vec = True
    else:
        print("No normalization stats found, using raw observations")
        env = make_env()
        use_vec = False

    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}")
    print(f"Running {args.episodes} episodes...\n")

    all_rewards = []
    for ep in range(args.episodes):
        if use_vec:
            obs = env.reset()
        else:
            obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if use_vec:
                obs, reward, dones, infos = env.step(action)
                total_reward += reward[0]
                steps += 1
                done = dones[0]
                info = infos[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

        ticks = info.get("episode_ticks", steps * args.frame_skip)
        all_rewards.append(total_reward)

        status = "done"
        print(f"  Episode {ep + 1}: reward={total_reward:+.2f}  steps={steps}  ticks={ticks}  ({status})")

    avg = sum(all_rewards) / len(all_rewards)
    best = max(all_rewards)
    worst = min(all_rewards)
    print(f"\n  Average: {avg:+.2f}  Best: {best:+.2f}  Worst: {worst:+.2f}")


if __name__ == "__main__":
    main()
