#!/usr/bin/env python3
"""Train TD-MPC from replay buffer and evaluate vs raw policy.

Progressive test: smoke test → small run → full evaluation.

Usage:
    python scripts/eval_tdmpc.py \
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
    rewards, step_times = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            t0 = time.perf_counter()
            action = action_fn(obs)
            step_times.append(time.perf_counter() - t0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
        if (ep + 1) % 5 == 0:
            print(f"  [{label}] ep {ep+1}/{n_episodes}: rew={ep_reward:.1f}")
    return np.array(rewards), np.array(step_times)


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
    from training.td_mpc import TDMPC

    gym_class = _import_gym_class(args.scenario)

    print("Loading assets...")
    buf = ReplayBuffer.load(args.buffer)
    print(f"  Buffer: {len(buf):,} transitions, obs_dim={buf.obs_dim}")

    tmp_env = gym_class(config_path=args.config, scenario=args.scenario,
                        frame_skip=args.frame_skip, max_steps=args.max_steps)
    policy, _ = load_model(args.policy, env=tmp_env)
    act_nvec = list(tmp_env.action_space.nvec)
    tmp_env.close()
    print(f"  Policy loaded, act_nvec={act_nvec}")

    # ==========================================
    # PHASE 1: Smoke test (200 train steps, 5 episodes)
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  PHASE 1: SMOKE TEST (200 train steps, 5 eval episodes)")
    print(f"{'='*60}\n")

    tdmpc = TDMPC(obs_dim=buf.obs_dim, act_nvec=act_nvec,
                   latent_dim=128, hidden=256, horizon=5,
                   n_candidates=128, mppi_iterations=3)
    tdmpc.train_from_buffer(buf, steps=200, batch_size=256, log_interval=50)

    env = gym_class(config_path=args.config, scenario=args.scenario,
                    frame_skip=args.frame_skip, max_steps=args.max_steps)
    raw_rew, _ = run_eval(env, lambda o: policy.predict(o, deterministic=False)[0], 5, "RAW")
    env.close()

    env = gym_class(config_path=args.config, scenario=args.scenario,
                    frame_skip=args.frame_skip, max_steps=args.max_steps)
    tdmpc_rew, tdmpc_times = run_eval(env, lambda o: tdmpc.plan(o).astype(int), 5, "TDMPC")
    env.close()

    print(f"\n  Phase 1: RAW={raw_rew.mean():.1f}, TDMPC={tdmpc_rew.mean():.1f}, "
          f"plan_time={np.mean(tdmpc_times)*1000:.1f}ms")

    if tdmpc_rew.mean() < 1.0:
        print("  [FAIL] TD-MPC reward < 1.0 — something is fundamentally wrong")
        return

    # ==========================================
    # PHASE 2: Medium run (2000 train steps, 20 episodes)
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  PHASE 2: MEDIUM RUN (2000 train steps, 20 eval episodes)")
    print(f"{'='*60}\n")

    tdmpc2 = TDMPC(obs_dim=buf.obs_dim, act_nvec=act_nvec,
                    latent_dim=128, hidden=256, horizon=5,
                    n_candidates=256, mppi_iterations=4)
    tdmpc2.train_from_buffer(buf, steps=2000, batch_size=256, log_interval=200)

    env = gym_class(config_path=args.config, scenario=args.scenario,
                    frame_skip=args.frame_skip, max_steps=args.max_steps)
    raw_rew2, _ = run_eval(env, lambda o: policy.predict(o, deterministic=False)[0], 20, "RAW")
    env.close()

    env = gym_class(config_path=args.config, scenario=args.scenario,
                    frame_skip=args.frame_skip, max_steps=args.max_steps)
    tdmpc_rew2, tdmpc_times2 = run_eval(env, lambda o: tdmpc2.plan(o).astype(int), 20, "TDMPC")
    env.close()

    print(f"\n  Phase 2: RAW={raw_rew2.mean():.1f}+/-{raw_rew2.std():.1f}, "
          f"TDMPC={tdmpc_rew2.mean():.1f}+/-{tdmpc_rew2.std():.1f}, "
          f"plan_time={np.mean(tdmpc_times2)*1000:.1f}ms")

    # Q-value discrimination check
    print("\n  Q-value discrimination:")
    z = tdmpc2._encode(buf.sample_states(1))
    for shoot in [0, 1]:
        for move in [0, 5, 8, 10]:
            test_act = np.array([[move, shoot, 0, 0]], dtype=np.float32)
            act_oh = tdmpc2._action_onehot(test_act)
            q = tdmpc2.q_function.min_q(z, act_oh).item()
            print(f"    move={move:2d} shoot={shoot}: Q={q:+.3f}")

    if tdmpc_rew2.mean() < raw_rew2.mean() * 0.5:
        print(f"\n  [WARN] TD-MPC reward is less than half of raw policy. "
              f"Planning may still be harmful. Continuing to Phase 3 anyway.")

    # ==========================================
    # PHASE 3: Full evaluation (5000 train steps, 50 episodes)
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  PHASE 3: FULL EVALUATION (5000 train steps, 50 eval episodes)")
    print(f"{'='*60}\n")

    tdmpc3 = TDMPC(obs_dim=buf.obs_dim, act_nvec=act_nvec,
                    latent_dim=128, hidden=256, horizon=5,
                    n_candidates=256, mppi_iterations=4,
                    policy_prior_weight=0.8)
    tdmpc3.train_from_buffer(buf, steps=5000, batch_size=256, log_interval=500)

    env = gym_class(config_path=args.config, scenario=args.scenario,
                    frame_skip=args.frame_skip, max_steps=args.max_steps)
    raw_rew3, _ = run_eval(env, lambda o: policy.predict(o, deterministic=False)[0], 50, "RAW")
    env.close()

    env = gym_class(config_path=args.config, scenario=args.scenario,
                    frame_skip=args.frame_skip, max_steps=args.max_steps)
    tdmpc_rew3, tdmpc_times3 = run_eval(env, lambda o: tdmpc3.plan(o).astype(int), 50, "TDMPC")
    env.close()

    # Also test policy-only from TD-MPC (no planning, just the learned policy head)
    env = gym_class(config_path=args.config, scenario=args.scenario,
                    frame_skip=args.frame_skip, max_steps=args.max_steps)
    def tdmpc_policy_only(obs):
        z = tdmpc3._encode(obs.reshape(1, -1) if obs.ndim == 1 else obs)
        return tdmpc3.policy.sample(z, temperature=0.5).squeeze(0).cpu().numpy().astype(int)
    policy_only_rew, _ = run_eval(env, tdmpc_policy_only, 50, "TDMPC_POLICY")
    env.close()

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\n  {'Condition':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, r in [("RAW_POLICY", raw_rew3), ("TDMPC_PLAN", tdmpc_rew3), ("TDMPC_POLICY", policy_only_rew)]:
        print(f"  {label:<20} {r.mean():>8.1f} {r.std():>8.1f} {r.min():>8.1f} {r.max():>8.1f}")

    print(f"\n  Plan time: {np.mean(tdmpc_times3)*1000:.1f}ms/step ({1000/np.mean(tdmpc_times3)/1000:.0f}kHz)")
    print(f"  TD-MPC train steps: {tdmpc3._train_steps}")

    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(tdmpc_rew3, raw_rew3, equal_var=False)
        d = (tdmpc_rew3.mean() - raw_rew3.mean()) / np.sqrt((raw_rew3.var() + tdmpc_rew3.var()) / 2)
        print(f"\n  TDMPC vs RAW: delta={tdmpc_rew3.mean()-raw_rew3.mean():+.1f}, p={p_val:.4f}, Cohen's d={d:+.2f}")
    except ImportError:
        pass

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        tdmpc3.save(os.path.join(args.save_dir, "tdmpc.pt"))
        print(f"\n  Saved to {args.save_dir}/")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
