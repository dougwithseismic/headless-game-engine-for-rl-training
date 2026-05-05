import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from ghostlobby_gym import GhostLobbyGym


def parse_args():
    p = argparse.ArgumentParser(description="GhostLobby RL Training")
    p.add_argument("--config", default="configs/1v1_deathmatch.json", help="Game config JSON")
    p.add_argument("--scenario", default="fps", help="Scenario name (fps, racing, moba)")
    p.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    p.add_argument("--frame-skip", type=int, default=4, help="Ticks per decision step")
    p.add_argument("--max-steps", type=int, default=2048, help="Max steps per episode")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    p.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    p.add_argument("--checkpoint-freq", type=int, default=50_000, help="Save checkpoint every N steps")
    p.add_argument("--eval-freq", type=int, default=25_000, help="Evaluate every N steps")
    p.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation")
    p.add_argument("--name", default=None, help="Experiment name (auto-generated if not set)")
    p.add_argument("--resume", default=None, help="Path to checkpoint .zip to resume from")
    return p.parse_args()


def make_run_dir(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = args.name or args.scenario
    run_name = f"{name}_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tensorboard"), exist_ok=True)
    return run_dir


def save_experiment_config(run_dir, args, env):
    config = {
        "name": args.name or args.scenario,
        "timestamp": datetime.now().isoformat(),
        "game_config": args.config,
        "scenario": args.scenario,
        "resumed_from": args.resume,
        "hyperparams": {
            "algorithm": "PPO",
            "total_timesteps": args.timesteps,
            "learning_rate": args.lr,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "ent_coef": args.ent_coef,
            "frame_skip": args.frame_skip,
            "max_episode_steps": args.max_steps,
        },
        "observation_space": list(env.observation_space.shape),
        "action_space": list(env.action_space.shape),
    }
    with open(os.path.join(run_dir, "experiment.json"), "w") as f:
        json.dump(config, f, indent=2)
    return config


def main():
    args = parse_args()
    run_dir = make_run_dir(args)

    print(f"Run directory: {run_dir}")

    train_env = GhostLobbyGym(
        config_path=args.config,
        scenario=args.scenario,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
    )

    eval_env = GhostLobbyGym(
        config_path=args.config,
        scenario=args.scenario,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
    )

    experiment = save_experiment_config(run_dir, args, train_env)
    print(f"Observation space: {train_env.observation_space.shape}")
    print(f"Action space: {train_env.action_space.shape}")

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=os.path.join(run_dir, "tensorboard"),
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=os.path.join(run_dir, "tensorboard"),
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="model",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    print(f"Training for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    print(f"Final model saved to {final_path}.zip")
    print(f"View dashboard: tensorboard --logdir {run_dir}/tensorboard")


if __name__ == "__main__":
    main()
