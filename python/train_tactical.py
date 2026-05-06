"""
PPO training for the TacticalDeathmatchScenario.

Usage:
    python train_tactical.py --config configs/tactical_open.json --timesteps 5000000
    python train_tactical.py --config configs/tactical_obstacles.json --timesteps 10000000
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv

from selfplay_gym import SelfPlayGym


def parse_args():
    p = argparse.ArgumentParser(description="Tactical Deathmatch Training")
    p.add_argument("--config", default="configs/tactical_open.json")
    p.add_argument("--timesteps", type=int, default=5_000_000)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=4096)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--swap-interval", type=int, default=500_000)
    p.add_argument("--scripted-warmup", type=int, default=1_000_000)
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    p.add_argument("--name", default="tactical")
    p.add_argument("--resume", default=None, help="Path to model .zip to resume from")
    p.add_argument("--live-view", action="store_true")
    p.add_argument("--live-port", type=int, default=3000)
    p.add_argument("--live-snapshot-freq", type=int, default=30_000)
    return p.parse_args()


def resolve_config(path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, path) if not os.path.isabs(path) else path


def make_env(config_path, frame_skip, max_steps):
    def _init():
        return SelfPlayGym(
            config_path=config_path,
            scenario="tactical",
            frame_skip=frame_skip,
            max_steps=max_steps,
        )
    return _init


class ThroughputCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.t0 = None

    def _on_training_start(self):
        self.t0 = time.perf_counter()

    def _on_step(self):
        if self.num_timesteps % 50_000 == 0 and self.t0:
            elapsed = time.perf_counter() - self.t0
            sps = self.num_timesteps / elapsed
            if self.verbose:
                print(f"  [{self.num_timesteps:>10,} steps] {sps:,.0f} steps/sec")
        return True


class SelfPlayCallback(BaseCallback):
    def __init__(self, swap_interval, scripted_warmup, train_env, max_history=20, verbose=1):
        super().__init__(verbose)
        self.swap_interval = swap_interval
        self.scripted_warmup = scripted_warmup
        self.train_env = train_env
        self.last_swap = 0
        self.swap_count = 0
        self.opponent_history = []
        self.max_history = max_history

    def _on_step(self):
        if self.num_timesteps < self.scripted_warmup:
            return True
        if self.num_timesteps - self.last_swap >= self.swap_interval:
            self._swap_opponent()
            self.last_swap = self.num_timesteps
        return True

    def _swap_opponent(self):
        import copy
        import random

        self.swap_count += 1
        snapshot = copy.deepcopy(self.model.policy)
        snapshot.set_training_mode(False)
        self.opponent_history.append(snapshot)
        if len(self.opponent_history) > self.max_history:
            self.opponent_history.pop(0)

        opponent = random.choice(self.opponent_history)
        inner_vec = self.train_env

        from train_selfplay import _NormalizedOpponent
        wrapped = _NormalizedOpponent(opponent, inner_vec)

        for i in range(inner_vec.num_envs):
            inner_vec.env_method("set_opponent", wrapped, indices=[i])

        if self.verbose:
            print(f"\n  [self-play] Swap #{self.swap_count} at step {self.num_timesteps:,}"
                  f" — pool size {len(self.opponent_history)}\n")


def main():
    args = parse_args()
    config_path = resolve_config(args.config)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", f"{args.name}_{timestamp}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tensorboard"), exist_ok=True)

    print(f"=== Tactical Deathmatch Training ===")
    print(f"Config: {args.config}")
    print(f"Run dir: {run_dir}")
    print(f"Envs: {args.n_envs} | Steps: {args.timesteps:,}")
    print(f"Warmup: {args.scripted_warmup:,} | Swap: {args.swap_interval:,}")

    env_fns = [make_env(config_path, args.frame_skip, args.max_steps)
               for _ in range(args.n_envs)]
    train_env = SubprocVecEnv(env_fns)

    eval_env = SubprocVecEnv([make_env(config_path, args.frame_skip, args.max_steps)])

    probe = SelfPlayGym(config_path, "tactical", args.frame_skip, args.max_steps)
    print(f"Obs: {probe.observation_space.shape} | Act: {probe.action_space.shape}")
    del probe

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=os.path.join(run_dir, "tensorboard"),
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
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

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=os.path.join(run_dir, "checkpoints"),
            name_prefix="model",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best_model"),
            log_path=os.path.join(run_dir, "eval_logs"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ),
        SelfPlayCallback(
            swap_interval=args.swap_interval,
            scripted_warmup=args.scripted_warmup,
            train_env=train_env,
            verbose=1,
        ),
        ThroughputCallback(verbose=1),
    ]

    if args.live_view:
        from live_view import LiveSnapshotCallback, start_live_view
        snapshot_path = os.path.join(run_dir, "live_snapshot")
        callbacks.append(LiveSnapshotCallback(snapshot_path, interval=args.live_snapshot_freq))
        start_live_view(
            config_path, "tactical", args.frame_skip, args.max_steps,
            snapshot_path, port=args.live_port,
        )

    with open(os.path.join(run_dir, "experiment.json"), "w") as f:
        json.dump({
            "config": args.config,
            "scenario": "tactical",
            "timesteps": args.timesteps,
            "n_envs": args.n_envs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "n_steps": args.n_steps,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "ent_coef": args.ent_coef,
            "frame_skip": args.frame_skip,
            "scripted_warmup": args.scripted_warmup,
            "swap_interval": args.swap_interval,
            "resume": args.resume,
        }, f, indent=2)

    print(f"\nStarting training...\n")
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks))
    elapsed = time.perf_counter() - t0

    model.save(os.path.join(run_dir, "final_model"))
    train_env.close()
    eval_env.close()

    print(f"\nDone. {args.timesteps:,} steps in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Model: {run_dir}/final_model.zip")
    print(f"Tensorboard: tensorboard --logdir {run_dir}/tensorboard")


if __name__ == "__main__":
    main()
