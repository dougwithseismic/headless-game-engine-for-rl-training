#!/usr/bin/env python3
"""Online world model training: Dreamer + TD-MPC with real env interaction.

The missing piece from all offline experiments. This script:
  1. Acts in the real environment using the world model policy
  2. Stores transitions in a growing replay buffer
  3. Trains the world model on real data every N steps
  4. For Dreamer: trains actor-critic on imagined trajectories
  5. For TD-MPC: plans using the learned model at each step

Usage:
    # Online Dreamer
    python scripts/train_online_worldmodel.py \
      --config configs/cs_lite/1v1_wide.json --algo dreamer \
      --steps 500000 --eval-freq 10000

    # Online TD-MPC
    python scripts/train_online_worldmodel.py \
      --config configs/cs_lite/1v1_wide.json --algo tdmpc \
      --steps 500000 --eval-freq 10000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def evaluate(env_factory, policy_fn, n_episodes=10):
    """Run eval episodes and return mean reward."""
    env = env_factory()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)
    env.close()
    return np.mean(rewards), np.std(rewards)


def train_dreamer_online(args, env_factory, act_nvec):
    from training.dreamer import Dreamer
    from training.replay_buffer import ReplayBuffer

    env = env_factory()
    obs_dim = env.observation_space.shape[0]

    dreamer = Dreamer(
        obs_dim=obs_dim, act_nvec=act_nvec,
        h_dim=args.h_dim, z_cats=args.z_cats, z_classes=args.z_classes,
        mlp_dim=args.mlp_dim, imagine_horizon=args.horizon,
    )
    buf = ReplayBuffer(capacity=args.buffer_cap, obs_dim=obs_dim, act_dim=len(act_nvec))

    obs, _ = env.reset()
    h_state, z_state = None, None
    episode_reward = 0.0
    episode_count = 0
    best_eval = -float('inf')

    print(f"\n{'='*60}")
    print(f"  ONLINE DREAMER TRAINING")
    print(f"  obs_dim={obs_dim}, act_nvec={act_nvec}")
    print(f"  Steps: {args.steps:,}, Train ratio: {args.train_ratio}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        action, h_state, z_state = dreamer.act(obs, h_state, z_state, temperature=0.5)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buf.add(obs, action.astype(np.float32), reward, next_obs, float(done))
        obs = next_obs
        episode_reward += reward

        if done:
            obs, _ = env.reset()
            h_state, z_state = None, None
            episode_count += 1
            episode_reward = 0.0

        if step >= args.warmup and step % args.train_freq == 0:
            for _ in range(args.train_ratio):
                batch = buf.sample(args.batch_size)
                dreamer.train_step(
                    batch["observations"], batch["actions"],
                    batch["rewards"], batch["next_observations"],
                    batch["dones"], seq_len=args.seq_len,
                )

        if step % args.eval_freq == 0:
            def dreamer_policy(o):
                a, _, _ = dreamer.act(o, temperature=0.3)
                return a
            mean_r, std_r = evaluate(env_factory, dreamer_policy, n_episodes=args.eval_eps)
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            print(
                f"  step {step:>8,} | eval={mean_r:.1f}+/-{std_r:.1f} | "
                f"buf={len(buf):,} | eps={episode_count} | "
                f"{sps:.0f} sps | {elapsed:.0f}s"
            )
            if mean_r > best_eval:
                best_eval = mean_r
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    dreamer.save(os.path.join(args.save_dir, "dreamer_best.pt"))

    env.close()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        dreamer.save(os.path.join(args.save_dir, "dreamer_final.pt"))
        buf.save(os.path.join(args.save_dir, "replay_buffer.npz"))

    final_mean, final_std = evaluate(
        env_factory, lambda o: dreamer.act(o, temperature=0.3)[0], n_episodes=20
    )
    print(f"\n  Final eval: {final_mean:.1f} +/- {final_std:.1f} (best={best_eval:.1f})")
    return dreamer, final_mean


def train_tdmpc_online(args, env_factory, act_nvec):
    from training.td_mpc_v2 import TDMPCv2
    from training.replay_buffer import ReplayBuffer

    env = env_factory()
    obs_dim = env.observation_space.shape[0]

    tdmpc = TDMPCv2(
        obs_dim=obs_dim, act_nvec=act_nvec,
        latent_dim=args.latent_dim, hidden=args.mlp_dim,
        horizon=args.horizon, n_candidates=args.n_candidates,
    )
    buf = ReplayBuffer(capacity=args.buffer_cap, obs_dim=obs_dim, act_dim=len(act_nvec))

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0
    best_eval = -float('inf')

    print(f"\n{'='*60}")
    print(f"  ONLINE TD-MPC v2 TRAINING")
    print(f"  obs_dim={obs_dim}, act_nvec={act_nvec}")
    print(f"  Steps: {args.steps:,}, Train ratio: {args.train_ratio}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        if step < args.warmup:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = tdmpc.plan(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action.astype(int))
        done = terminated or truncated

        buf.add(obs, action.astype(np.float32), reward, next_obs, float(done))
        obs = next_obs
        episode_reward += reward

        if done:
            obs, _ = env.reset()
            episode_count += 1
            episode_reward = 0.0

        if len(buf) >= args.warmup and step % args.train_freq == 0:
            H = tdmpc.horizon
            data = buf.all_data()
            rng = np.random.default_rng()
            for _ in range(args.train_ratio):
                idx = rng.integers(0, len(buf) - H, size=args.batch_size)
                obs_seq = np.stack([data["observations"][idx + t] for t in range(H)])
                act_seq = np.stack([data["actions"][idx + t] for t in range(H)])
                rew_seq = np.stack([data["rewards"][idx + t] for t in range(H)])
                next_seq = np.stack([data["next_observations"][idx + t] for t in range(H)])
                done_seq = np.stack([data["dones"][idx + t] for t in range(H)])
                tdmpc.train_step(obs_seq, act_seq, rew_seq, next_seq, done_seq)

        if step % args.eval_freq == 0:
            mean_r, std_r = evaluate(
                env_factory, lambda o: tdmpc.plan(o).astype(int), n_episodes=args.eval_eps
            )
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            print(
                f"  step {step:>8,} | eval={mean_r:.1f}+/-{std_r:.1f} | "
                f"buf={len(buf):,} | eps={episode_count} | "
                f"{sps:.0f} sps | {elapsed:.0f}s"
            )
            if mean_r > best_eval:
                best_eval = mean_r
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    tdmpc.save(os.path.join(args.save_dir, "tdmpc_best.pt"))

    env.close()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        tdmpc.save(os.path.join(args.save_dir, "tdmpc_final.pt"))
        buf.save(os.path.join(args.save_dir, "replay_buffer.npz"))

    final_mean, final_std = evaluate(
        env_factory, lambda o: tdmpc.plan(o).astype(int), n_episodes=20
    )
    print(f"\n  Final eval: {final_mean:.1f} +/- {final_std:.1f} (best={best_eval:.1f})")
    return tdmpc, final_mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--scenario", default="cs_lite")
    p.add_argument("--algo", required=True, choices=["dreamer", "tdmpc"])
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--train-ratio", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-cap", type=int, default=500000)
    p.add_argument("--eval-freq", type=int, default=10000)
    p.add_argument("--eval-eps", type=int, default=5)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2048)
    p.add_argument("--save-dir", default=None)

    p.add_argument("--h-dim", type=int, default=512)
    p.add_argument("--z-cats", type=int, default=32)
    p.add_argument("--z-classes", type=int, default=32)
    p.add_argument("--mlp-dim", type=int, default=256)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--n-candidates", type=int, default=256)
    args = p.parse_args()

    from training.ppo_trainer import _import_gym_class
    gym_class = _import_gym_class(args.scenario)

    def env_factory():
        return gym_class(
            config_path=args.config, scenario=args.scenario,
            frame_skip=args.frame_skip, max_steps=args.max_steps,
        )

    tmp = env_factory()
    act_nvec = list(tmp.action_space.nvec)
    tmp.close()

    if args.algo == "dreamer":
        train_dreamer_online(args, env_factory, act_nvec)
    else:
        train_tdmpc_online(args, env_factory, act_nvec)


if __name__ == "__main__":
    main()
