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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from ghostlobby_gym import GhostLobbyGym


class ThroughputCallback(BaseCallback):
    def __init__(self, n_envs, log_interval=1, verbose=1):
        super().__init__(verbose)
        self.n_envs = n_envs
        self.log_interval = log_interval
        self.rollout_start = None
        self.rollout_end = None
        self.iteration = 0
        self.history = []

    def _on_training_start(self):
        self.rollout_start = time.perf_counter()

    def _on_rollout_start(self):
        now = time.perf_counter()
        if self.rollout_end is not None:
            update_time = now - self.rollout_end
            self._log_iteration(update_time)
        self.rollout_start = now

    def _on_rollout_end(self):
        now = time.perf_counter()
        self._last_rollout_time = now - self.rollout_start
        n_steps = self.model.n_steps
        self._last_total_steps = n_steps * self.n_envs
        self._last_rollout_sps = self._last_total_steps / self._last_rollout_time if self._last_rollout_time > 0 else 0
        self.rollout_end = now

    def _on_step(self):
        return True

    def _log_iteration(self, update_time):
        self.iteration += 1
        rollout_time = self._last_rollout_time
        rollout_sps = self._last_rollout_sps
        total_steps = self._last_total_steps
        total_time = rollout_time + update_time
        effective_sps = total_steps / total_time if total_time > 0 else 0
        env_pct = (rollout_time / total_time * 100) if total_time > 0 else 0
        update_pct = 100 - env_pct

        self.history.append({
            "iteration": self.iteration,
            "rollout_sps": rollout_sps,
            "effective_sps": effective_sps,
            "rollout_time": rollout_time,
            "update_time": update_time,
            "env_pct": env_pct,
        })

        if self.verbose and self.iteration % self.log_interval == 0:
            print(
                f"  [throughput] iter {self.iteration:>4} | "
                f"rollout: {rollout_sps:>8,.0f} sps ({rollout_time:.2f}s, {env_pct:.0f}%) | "
                f"update: {update_time:.2f}s ({update_pct:.0f}%) | "
                f"effective: {effective_sps:>8,.0f} sps"
            )

    def _on_training_end(self):
        if self.rollout_end is not None:
            self._log_iteration(time.perf_counter() - self.rollout_end)
        self.summary()

    def summary(self):
        if not self.history:
            return
        import numpy as np
        rollout_rates = [h["rollout_sps"] for h in self.history]
        effective_rates = [h["effective_sps"] for h in self.history]
        env_pcts = [h["env_pct"] for h in self.history]
        print()
        print("=" * 65)
        print("Throughput Summary")
        print("=" * 65)
        print(f"  Iterations:         {len(self.history)}")
        print(f"  Rollout sps:        {np.mean(rollout_rates):>8,.0f} avg / {np.max(rollout_rates):>8,.0f} peak")
        print(f"  Effective sps:      {np.mean(effective_rates):>8,.0f} avg / {np.max(effective_rates):>8,.0f} peak")
        print(f"  Time in env:        {np.mean(env_pcts):.0f}% avg")
        print(f"  Time in update:     {100 - np.mean(env_pcts):.0f}% avg")
        print("=" * 65)


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
    p.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant", help="LR schedule")
    p.add_argument("--clip-schedule", choices=["constant", "linear"], default="constant", help="Clip range schedule")
    p.add_argument("--checkpoint-freq", type=int, default=50_000, help="Save checkpoint every N steps")
    p.add_argument("--eval-freq", type=int, default=25_000, help="Evaluate every N steps")
    p.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation")
    p.add_argument("--name", default=None, help="Experiment name (auto-generated if not set)")
    p.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    p.add_argument("--resume", default=None, help="Path to checkpoint .zip to resume from")
    p.add_argument("--resume-norm", default=None, help="Path to vec_normalize.pkl to resume normalization")
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
            "lr_schedule": args.lr_schedule,
            "clip_schedule": args.clip_schedule,
            "frame_skip": args.frame_skip,
            "max_episode_steps": args.max_steps,
            "n_envs": args.n_envs,
        },
        "observation_space": list(env.observation_space.shape),
        "action_space": list(env.action_space.shape),
    }
    with open(os.path.join(run_dir, "experiment.json"), "w") as f:
        json.dump(config, f, indent=2)
    return config


def make_env(config_path, scenario, frame_skip, max_steps):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_config = os.path.join(project_root, config_path) if not os.path.isabs(config_path) else config_path
    def _init():
        return GhostLobbyGym(
            config_path=abs_config,
            scenario=scenario,
            frame_skip=frame_skip,
            max_steps=max_steps,
        )
    return _init


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    run_dir = make_run_dir(args)

    print(f"Run directory: {run_dir}")
    print(f"Parallel environments: {args.n_envs}")

    env_fns = [make_env(args.config, args.scenario, args.frame_skip, args.max_steps)
               for _ in range(args.n_envs)]

    if args.resume_norm:
        print(f"Resuming normalization from: {args.resume_norm}")
        train_env = VecNormalize.load(args.resume_norm, SubprocVecEnv(env_fns))
        train_env.training = True
        train_env.norm_reward = True

        eval_env = VecNormalize.load(
            args.resume_norm,
            SubprocVecEnv([make_env(args.config, args.scenario, args.frame_skip, args.max_steps)]),
        )
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        train_env = VecNormalize(
            SubprocVecEnv(env_fns),
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        eval_env = VecNormalize(
            SubprocVecEnv([make_env(args.config, args.scenario, args.frame_skip, args.max_steps)]),
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

    probe_env = GhostLobbyGym(
        config_path=args.config,
        scenario=args.scenario,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
    )
    experiment = save_experiment_config(run_dir, args, probe_env)
    print(f"Observation space: {probe_env.observation_space.shape}")
    print(f"Action space: {probe_env.action_space.shape}")
    del probe_env

    rollout_buffer_size = args.n_steps * args.n_envs
    print(f"Rollout buffer: {args.n_steps} steps/env × {args.n_envs} envs = {rollout_buffer_size:,} total")

    lr = args.lr
    if args.lr_schedule == "linear":
        base_lr = args.lr
        lr = lambda progress: base_lr * progress

    clip_range = 0.2
    if args.clip_schedule == "linear":
        clip_range = lambda progress: 0.2 * progress

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
            learning_rate=lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=clip_range,
            ent_coef=args.ent_coef,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="model",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    throughput_cb = ThroughputCallback(n_envs=args.n_envs, verbose=1)
    callbacks = CallbackList([checkpoint_cb, eval_cb, throughput_cb])

    print(f"Training for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    train_env.save(os.path.join(run_dir, "vec_normalize.pkl"))

    train_env.close()
    eval_env.close()

    print(f"Final model saved to {final_path}.zip")
    print(f"View dashboard: tensorboard --logdir {run_dir}/tensorboard")


if __name__ == "__main__":
    main()
