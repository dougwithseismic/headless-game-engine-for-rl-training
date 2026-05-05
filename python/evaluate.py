import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO

from ghostlobby_gym import GhostLobbyGym


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained GhostLobby agent")
    p.add_argument("--model", required=True, help="Path to model .zip file")
    p.add_argument("--config", default="configs/1v1_deathmatch.json", help="Game config")
    p.add_argument("--scenario", default="fps", help="Scenario name")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    p.add_argument("--frame-skip", type=int, default=4, help="Ticks per step")
    p.add_argument("--max-steps", type=int, default=2048, help="Max steps per episode")
    return p.parse_args()


def main():
    args = parse_args()

    env = GhostLobbyGym(
        config_path=args.config,
        scenario=args.scenario,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
    )

    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}")
    print(f"Running {args.episodes} episodes...\n")

    all_rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        ticks = info.get("episode_ticks", steps * args.frame_skip)
        all_rewards.append(total_reward)

        status = "terminated" if terminated else "truncated"
        print(f"  Episode {ep + 1}: reward={total_reward:+.2f}  steps={steps}  ticks={ticks}  ({status})")

    avg = sum(all_rewards) / len(all_rewards)
    best = max(all_rewards)
    worst = min(all_rewards)
    print(f"\n  Average: {avg:+.2f}  Best: {best:+.2f}  Worst: {worst:+.2f}")


if __name__ == "__main__":
    main()
