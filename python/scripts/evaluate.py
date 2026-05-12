#!/usr/bin/env python3
"""Evaluate a trained model.

Usage:
    python scripts/evaluate.py --model runs/arena3d_p1/best_model.zip --config configs/arena3d/phase1_aim_targets.json --episodes 10
    python scripts/evaluate.py --model runs/arena3d_p1/best_model.zip --config configs/arena3d/phase1_aim_targets.json --scenario tactical --episodes 20
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser(description="Evaluate trained model")
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--scenario", default="arena3d")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--phase", type=int, default=None)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2048)
    args = p.parse_args()

    from training.utils import load_model, predict_with_state
    from training.ppo_trainer import _import_gym_class

    gym_class = _import_gym_class(args.scenario)
    env = gym_class(
        config_path=args.config,
        scenario=args.scenario,
        phase=args.phase,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
    )
    model, is_recurrent = load_model(args.model, env=env)

    if is_recurrent:
        print("Loaded recurrent (LSTM) model")
    else:
        print("Loaded feed-forward model")

    import numpy as np

    rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_r = 0
        done = False
        steps = 0
        state = None
        episode_start = True

        while not done:
            if is_recurrent:
                action, state = predict_with_state(
                    model, obs, state=state,
                    episode_start=episode_start,
                    deterministic=args.deterministic,
                )
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)

            obs, r, term, trunc, info = env.step(action)
            total_r += r
            done = term or trunc
            steps += 1
            episode_start = False

        rewards.append(total_r)
        print(f"  Episode {ep+1}: reward={total_r:.2f} steps={steps}")

    print(f"\nMean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Min: {np.min(rewards):.2f}  Max: {np.max(rewards):.2f}")
    env.close()


if __name__ == "__main__":
    main()
