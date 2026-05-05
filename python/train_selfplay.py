"""
Self-play PPO training for GhostLobby.

Every --swap-interval steps, the current policy is frozen and becomes the opponent
for the next training phase. Early phases use scripted AI as the opponent.
"""

import argparse
import copy
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

from selfplay_gym import SelfPlayGym


def parse_args():
    p = argparse.ArgumentParser(description="GhostLobby Self-Play Training")
    p.add_argument("--config", default="configs/1v1_deathmatch.json")
    p.add_argument("--scenario", default="fps")
    p.add_argument("--timesteps", type=int, default=20_000_000)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=4096)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--swap-interval", type=int, default=500_000,
                   help="Steps between opponent policy swaps")
    p.add_argument("--scripted-warmup", type=int, default=1_000_000,
                   help="Steps of training against scripted AI before self-play begins")
    p.add_argument("--eval-freq", type=int, default=500_000)
    p.add_argument("--checkpoint-freq", type=int, default=2_000_000)
    p.add_argument("--name", default="selfplay")
    p.add_argument("--resume", default=None, help="Path to model .zip to resume from")
    p.add_argument("--resume-norm", default=None, help="Path to vec_normalize.pkl to resume normalization stats")
    p.add_argument("--no-normalize", action="store_true", help="Disable VecNormalize (use with pre-normalized obs)")
    return p.parse_args()


def resolve_config(path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, path) if not os.path.isabs(path) else path


def make_env(config_path, scenario, frame_skip, max_steps):
    def _init():
        return SelfPlayGym(
            config_path=config_path,
            scenario=scenario,
            frame_skip=frame_skip,
            max_steps=max_steps,
        )
    return _init


class SelfPlayCallback(BaseCallback):
    def __init__(self, swap_interval, scripted_warmup, train_env, verbose=1):
        super().__init__(verbose)
        self.swap_interval = swap_interval
        self.scripted_warmup = scripted_warmup
        self.train_env = train_env
        self.last_swap = 0
        self.swap_count = 0
        self.opponent_history = []

    def _on_step(self):
        steps = self.num_timesteps

        if steps < self.scripted_warmup:
            return True

        if steps - self.last_swap >= self.swap_interval:
            self._swap_opponent()
            self.last_swap = steps

        return True

    def _swap_opponent(self):
        self.swap_count += 1
        opponent = copy.deepcopy(self.model.policy)
        opponent.set_training_mode(False)

        vec_norm = self.train_env
        inner_vec = vec_norm.venv if hasattr(vec_norm, 'venv') else vec_norm

        wrapped = _NormalizedOpponent(opponent, vec_norm)

        for i in range(inner_vec.num_envs):
            inner_vec.env_method("set_opponent", wrapped, indices=[i])

        if self.verbose:
            print(f"\n  [self-play] Swap #{self.swap_count} at step {self.num_timesteps:,}"
                  f" — opponent updated to current policy\n")


class _NormalizedOpponent:
    """Wraps a frozen policy (optionally with VecNormalize stats) to produce actions."""
    def __init__(self, policy, env_wrapper):
        self.policy = policy
        self.obs_rms = copy.deepcopy(env_wrapper.obs_rms) if hasattr(env_wrapper, 'obs_rms') else None
        self.clip_obs = getattr(env_wrapper, 'clip_obs', 10.0)
        self.epsilon = getattr(env_wrapper, 'epsilon', 1e-8)

    def predict(self, obs, deterministic=False):
        import numpy as np
        if self.obs_rms is not None:
            obs = np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                -self.clip_obs, self.clip_obs
            ).astype(np.float32)
        import torch
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(self.policy.device)
            dist = self.policy.get_distribution(obs_tensor)
            action = dist.sample() if not deterministic else dist.mode()
        return action.cpu().numpy().squeeze(0), None


def main():
    args = parse_args()
    config_path = resolve_config(args.config)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", f"{args.name}_{timestamp}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tensorboard"), exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Self-play swap interval: {args.swap_interval:,} steps")
    print(f"Scripted AI warmup: {args.scripted_warmup:,} steps")

    env_fns = [make_env(config_path, args.scenario, args.frame_skip, args.max_steps)
               for _ in range(args.n_envs)]

    if args.no_normalize:
        print("VecNormalize disabled (pre-normalized observations)")
        train_env = SubprocVecEnv(env_fns)
        eval_env = SubprocVecEnv([make_env(config_path, args.scenario, args.frame_skip, args.max_steps)])
    elif args.resume_norm:
        print(f"Resuming normalization from: {args.resume_norm}")
        train_env = VecNormalize.load(args.resume_norm, SubprocVecEnv(env_fns))
        train_env.training = True
        train_env.norm_reward = True

        eval_sub = SubprocVecEnv([make_env(config_path, args.scenario, args.frame_skip, args.max_steps)])
        eval_env = VecNormalize.load(args.resume_norm, eval_sub)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        train_env = VecNormalize(
            SubprocVecEnv(env_fns),
            norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
        )
        eval_env = VecNormalize(
            SubprocVecEnv([make_env(config_path, args.scenario, args.frame_skip, args.max_steps)]),
            norm_obs=True, norm_reward=False, clip_obs=10.0,
        )

    probe_env = SelfPlayGym(config_path, args.scenario, args.frame_skip, args.max_steps)
    print(f"Observation space: {probe_env.observation_space.shape}")
    print(f"Action space: {probe_env.action_space.shape}")
    del probe_env

    if args.resume:
        print(f"Resuming model from: {args.resume}")
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
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    selfplay_cb = SelfPlayCallback(
        swap_interval=args.swap_interval,
        scripted_warmup=args.scripted_warmup,
        train_env=train_env,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb, selfplay_cb])

    print(f"\nPhase 1: Scripted AI warmup ({args.scripted_warmup:,} steps)")
    print(f"Phase 2: Self-play with swaps every {args.swap_interval:,} steps")
    print(f"Total: {args.timesteps:,} steps\n")

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    elapsed = time.perf_counter() - t0

    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    if hasattr(train_env, 'save'):
        train_env.save(os.path.join(run_dir, "vec_normalize.pkl"))

    train_env.close()
    eval_env.close()

    print(f"\nFinal model saved to {final_path}.zip")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Self-play swaps: {selfplay_cb.swap_count}")
    print(f"View dashboard: tensorboard --logdir {run_dir}/tensorboard")


if __name__ == "__main__":
    main()
