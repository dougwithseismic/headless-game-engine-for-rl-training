#!/usr/bin/env python3
"""Evaluate MPC planning vs raw policy on real CsLite episodes.

Tests multiple MPC strategies to isolate where lookahead helps:
  1. RAW: trained policy, no planning
  2. TACTICAL: policy controls movement, MPC controls combat (shoot/reload/use)
  3. GATED: use MPC only when it predicts significantly better outcome
  4. FULL: MPC controls everything (expected to fail — destroys coherent movement)

Usage:
    python scripts/eval_mpc.py \
      --policy runs/.../best_model.zip \
      --dynamics runs/world_model_expert/dynamics_ensemble.pt \
      --config configs/cs_lite/1v1_wide.json
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def run_episodes(env, action_fn, n_episodes, label):
    """Run episodes collecting rewards and timing."""
    rewards, lengths, step_times = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward, ep_steps = 0.0, 0

        while not done:
            t0 = time.perf_counter()
            action = action_fn(obs)
            step_times.append(time.perf_counter() - t0)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

        rewards.append(ep_reward)
        lengths.append(ep_steps)
        if (ep + 1) % 10 == 0:
            print(f"  [{label}] ep {ep+1}/{n_episodes}: rew={ep_reward:.1f}, steps={ep_steps}")

    return np.array(rewards), np.array(lengths), np.array(step_times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--dynamics", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--scenario", default="cs_lite")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--candidates", type=int, default=256)
    p.add_argument("--gate-threshold", type=float, default=0.05,
                   help="Minimum predicted advantage to override policy (GATED mode)")
    args = p.parse_args()

    from training.ppo_trainer import _import_gym_class
    from training.utils import load_model
    from training.dynamics_model import DynamicsEnsemble
    from training.mpc_planner import MPCPlanner

    gym_class = _import_gym_class(args.scenario)

    print("Loading assets...")
    tmp_env = gym_class(config_path=args.config, scenario=args.scenario,
                        frame_skip=args.frame_skip, max_steps=args.max_steps)
    policy, _ = load_model(args.policy, env=tmp_env)
    tmp_env.close()

    ensemble = DynamicsEnsemble.load(args.dynamics)
    act_nvec = list(ensemble.act_nvec)

    planner = MPCPlanner(
        ensemble, act_nvec=act_nvec, horizon=args.horizon,
        n_candidates=args.candidates, n_elites=32, cem_iterations=2, gamma=0.99,
    )

    print(f"\n{'='*70}")
    print(f"  MPC EXPERIMENT")
    print(f"  Policy: {args.policy}")
    print(f"  Dynamics: {args.dynamics}")
    print(f"  Horizon: {args.horizon}, Candidates: {args.candidates}")
    print(f"  Episodes per condition: {args.episodes}")
    print(f"  Gate threshold: {args.gate_threshold}")
    print(f"{'='*70}")

    conditions = {}

    # --- 1. RAW POLICY ---
    print(f"\n--- RAW POLICY (baseline) ---")
    env = gym_class(config_path=args.config, scenario=args.scenario,
                    frame_skip=args.frame_skip, max_steps=args.max_steps)
    def raw_fn(obs):
        a, _ = policy.predict(obs, deterministic=False)
        return a
    rew, lens, times = run_episodes(env, raw_fn, args.episodes, "RAW")
    env.close()
    conditions["RAW"] = rew
    print(f"  Result: {rew.mean():.1f} +/- {rew.std():.1f}")

    # --- 2. TACTICAL MPC ---
    # Policy controls movement (head 0), MPC controls combat heads (1,2,3)
    print(f"\n--- TACTICAL MPC (policy=movement, MPC=combat) ---")
    env = gym_class(config_path=args.config, scenario=args.scenario,
                    frame_skip=args.frame_skip, max_steps=args.max_steps)

    def tactical_fn(obs):
        policy_action, _ = policy.predict(obs, deterministic=False)
        policy_action = np.asarray(policy_action, dtype=np.float32)

        # MPC plans all heads
        mpc_action = planner.plan(obs)

        # Use policy for movement (head 0), MPC for combat (heads 1,2,3)
        combined = policy_action.copy()
        combined[1:] = mpc_action[1:]
        return combined.astype(int)

    rew, lens, times = run_episodes(env, tactical_fn, args.episodes, "TACTICAL")
    env.close()
    conditions["TACTICAL"] = rew
    print(f"  Result: {rew.mean():.1f} +/- {rew.std():.1f}")
    print(f"  Decision time: {np.mean(times)*1000:.1f}ms/step")

    # --- 3. GATED MPC ---
    # Use MPC only when predicted return is significantly better than policy
    print(f"\n--- GATED MPC (override only when MPC >> policy, threshold={args.gate_threshold}) ---")
    env = gym_class(config_path=args.config, scenario=args.scenario,
                    frame_skip=args.frame_skip, max_steps=args.max_steps)

    gate_stats = {"overrides": 0, "total": 0}

    def gated_fn(obs):
        policy_action, _ = policy.predict(obs, deterministic=False)
        policy_action = np.asarray(policy_action, dtype=np.float32)

        mpc_action = planner.plan(obs)

        obs_2d = obs.reshape(1, -1) if obs.ndim == 1 else obs
        p_act = policy_action.reshape(1, -1)
        m_act = mpc_action.reshape(1, -1)

        _, p_rew, _ = ensemble.predict(obs_2d, p_act)
        _, m_rew, _ = ensemble.predict(obs_2d, m_act)

        gate_stats["total"] += 1
        if m_rew[0] - p_rew[0] > args.gate_threshold:
            gate_stats["overrides"] += 1
            return mpc_action.astype(int)
        return policy_action.astype(int)

    rew, lens, times = run_episodes(env, gated_fn, args.episodes, "GATED")
    env.close()
    conditions["GATED"] = rew
    override_pct = 100 * gate_stats["overrides"] / max(gate_stats["total"], 1)
    print(f"  Result: {rew.mean():.1f} +/- {rew.std():.1f}")
    print(f"  Override rate: {override_pct:.1f}% ({gate_stats['overrides']}/{gate_stats['total']})")
    print(f"  Decision time: {np.mean(times)*1000:.1f}ms/step")

    # --- 4. FULL MPC (control experiment — expected to fail) ---
    print(f"\n--- FULL MPC (MPC controls everything — control experiment) ---")
    env = gym_class(config_path=args.config, scenario=args.scenario,
                    frame_skip=args.frame_skip, max_steps=args.max_steps)

    def full_mpc_fn(obs):
        return planner.plan(obs).astype(int)

    rew, lens, times = run_episodes(env, full_mpc_fn, args.episodes, "FULL_MPC")
    env.close()
    conditions["FULL_MPC"] = rew
    print(f"  Result: {rew.mean():.1f} +/- {rew.std():.1f}")
    print(f"  Decision time: {np.mean(times)*1000:.1f}ms/step")

    # --- FINAL COMPARISON ---
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")

    raw_mean = conditions["RAW"].mean()
    print(f"\n  {'Condition':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'vs RAW':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for label, r in conditions.items():
        delta = r.mean() - raw_mean
        delta_str = f"{delta:+.1f}" if label != "RAW" else "---"
        print(f"  {label:<15} {r.mean():>8.1f} {r.std():>8.1f} {r.min():>8.1f} {r.max():>8.1f} {delta_str:>10}")

    try:
        from scipy import stats
        print(f"\n  --- Statistical Significance (Welch's t-test vs RAW) ---")
        for label, r in conditions.items():
            if label == "RAW":
                continue
            t_stat, p_val = stats.ttest_ind(r, conditions["RAW"], equal_var=False)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            pooled_std = np.sqrt((conditions["RAW"].var() + r.var()) / 2)
            d = (r.mean() - raw_mean) / pooled_std if pooled_std > 0 else 0
            print(f"  {label:<15} delta={r.mean()-raw_mean:+.1f}  p={p_val:.4f} {sig}  Cohen's d={d:+.2f}")
    except ImportError:
        print("  (scipy not available for statistical tests)")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
