#!/usr/bin/env python3
"""Train Dreamer from replay buffer and evaluate vs raw policy.

Progressive: smoke → medium → full.

Usage:
    python scripts/eval_dreamer.py \
      --policy runs/.../best_model.zip \
      --buffer runs/world_model_expert/replay_buffer.npz \
      --config configs/cs_lite/1v1_wide.json
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def run_eval(env, action_fn, n_episodes, label):
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            action = action_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
        if (ep + 1) % 5 == 0:
            print(f"  [{label}] ep {ep+1}/{n_episodes}: rew={ep_reward:.1f}")
    return np.array(rewards)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--buffer", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--scenario", default="cs_lite")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2048)
    p.add_argument("--save-dir", default=None)
    args = p.parse_args()

    from training.ppo_trainer import _import_gym_class
    from training.utils import load_model
    from training.replay_buffer import ReplayBuffer
    from training.dreamer import Dreamer

    gym_class = _import_gym_class(args.scenario)

    print("Loading assets...")
    buf = ReplayBuffer.load(args.buffer)
    print(f"  Buffer: {len(buf):,} transitions")

    tmp_env = gym_class(config_path=args.config, scenario=args.scenario,
                        frame_skip=args.frame_skip, max_steps=args.max_steps)
    policy, _ = load_model(args.policy, env=tmp_env)
    act_nvec = list(tmp_env.action_space.nvec)
    tmp_env.close()

    phases = [
        ("PHASE 1: SMOKE", 500, 10),
        ("PHASE 2: MEDIUM", 2000, 20),
        ("PHASE 3: FULL", 5000, 50),
    ]

    for phase_name, train_steps, eval_eps in phases:
        print(f"\n{'='*60}")
        print(f"  {phase_name} ({train_steps} train steps, {eval_eps} eval episodes)")
        print(f"{'='*60}\n")

        dreamer = Dreamer(
            obs_dim=buf.obs_dim, act_nvec=act_nvec,
            h_dim=512, z_cats=32, z_classes=32, mlp_dim=256,
            imagine_horizon=15, gamma=0.997,
        )

        t0 = time.perf_counter()
        dreamer.train_from_buffer(buf, steps=train_steps, batch_size=512,
                                   seq_len=16, log_interval=train_steps//5)
        train_time = time.perf_counter() - t0
        print(f"  Training time: {train_time:.1f}s")

        # Eval raw policy
        env = gym_class(config_path=args.config, scenario=args.scenario,
                        frame_skip=args.frame_skip, max_steps=args.max_steps)
        raw_rew = run_eval(env, lambda o: policy.predict(o, deterministic=False)[0], eval_eps, "RAW")
        env.close()

        # Eval Dreamer (stateless predict)
        env = gym_class(config_path=args.config, scenario=args.scenario,
                        frame_skip=args.frame_skip, max_steps=args.max_steps)
        dreamer_rew = run_eval(env, lambda o: dreamer.predict(o, deterministic=False)[0], eval_eps, "DREAMER")
        env.close()

        # Eval Dreamer (stateful — maintains GRU state across steps)
        env = gym_class(config_path=args.config, scenario=args.scenario,
                        frame_skip=args.frame_skip, max_steps=args.max_steps)
        h_state, z_state = [None], [None]
        def dreamer_stateful(obs):
            action, h_state[0], z_state[0] = dreamer.act(obs, h_state[0], z_state[0], temperature=0.3)
            return action
        # Need to reset state on env reset
        orig_reset = env.reset
        def reset_wrapper(*a, **k):
            h_state[0], z_state[0] = None, None
            return orig_reset(*a, **k)
        env.reset = reset_wrapper
        stateful_rew = run_eval(env, dreamer_stateful, eval_eps, "DREAMER_STATE")
        env.close()

        print(f"\n  {phase_name} Results:")
        print(f"  {'Condition':<20} {'Mean':>8} {'Std':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8}")
        for label, r in [("RAW_POLICY", raw_rew), ("DREAMER_STATELESS", dreamer_rew), ("DREAMER_STATEFUL", stateful_rew)]:
            print(f"  {label:<20} {r.mean():>8.1f} {r.std():>8.1f}")

        if phase_name.startswith("PHASE 3") and args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            dreamer.save(os.path.join(args.save_dir, "dreamer.pt"))
            print(f"\n  Saved to {args.save_dir}/")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
