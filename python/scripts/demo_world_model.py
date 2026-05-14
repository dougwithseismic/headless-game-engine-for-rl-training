#!/usr/bin/env python3
"""End-to-end demonstration of the Dyna world model.

Collects transitions from the game environment (scripted AI or random),
trains a dynamics ensemble, evaluates prediction accuracy at multiple
horizons, and generates imagined rollouts.

Usage:
    # With game engine (collects real transitions from scripted AI):
    python scripts/demo_world_model.py --config configs/cs_lite/1v1_wide.json --scenario cs_lite

    # Without game engine (synthetic data, for testing the pipeline):
    python scripts/demo_world_model.py --synthetic

    # Load existing replay buffer:
    python scripts/demo_world_model.py --load-buffer data/replay_buffer.npz
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def collect_from_engine(config_path, scenario, n_episodes, frame_skip, max_steps, policy_path=None):
    """Collect transitions from the game engine.

    If policy_path is given, loads the trained model and uses it to act.
    Otherwise uses random actions.
    """
    from training.ppo_trainer import _import_gym_class

    gym_class = _import_gym_class(scenario)
    env = gym_class(
        config_path=config_path,
        scenario=scenario,
        frame_skip=frame_skip,
        max_steps=max_steps,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else len(env.action_space.nvec)
    act_nvec = list(env.action_space.nvec) if hasattr(env.action_space, 'nvec') else None

    policy = None
    policy_state = None
    is_recurrent = False
    if policy_path:
        from training.utils import load_model
        policy, is_recurrent = load_model(policy_path, env=env)
        print(f"  Loaded policy from {policy_path} (recurrent={is_recurrent})")

    from training.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(capacity=n_episodes * max_steps, obs_dim=obs_dim, act_dim=act_dim)

    total_transitions = 0
    total_reward = 0.0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        if is_recurrent:
            policy_state = None
            episode_start = True

        while not done:
            if policy is not None:
                if is_recurrent:
                    from training.utils import predict_with_state
                    action, policy_state = predict_with_state(
                        policy, obs, state=policy_state,
                        episode_start=episode_start, deterministic=False,
                    )
                    episode_start = False
                else:
                    action, _ = policy.predict(obs, deterministic=False)
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(obs, np.asarray(action, dtype=np.float32), reward, next_obs, float(done))
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            total_transitions += 1

        total_reward += ep_reward
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: {ep_steps} steps, reward={ep_reward:.2f}")

    env.close()
    mean_reward = total_reward / n_episodes
    print(f"\n  Collected {total_transitions:,} transitions from {n_episodes} episodes")
    print(f"  Mean episode reward: {mean_reward:.2f}")
    if policy_path:
        print(f"  Policy: {policy_path}")

    return buffer, obs_dim, act_dim, act_nvec


def generate_synthetic_data(n_transitions=10000, obs_dim=250, act_nvec=None):
    """Generate synthetic transitions with learnable dynamics."""
    if act_nvec is None:
        act_nvec = [12, 2, 2, 3]
    act_dim = len(act_nvec)

    from training.replay_buffer import ReplayBuffer
    rng = np.random.RandomState(42)

    obs = rng.randn(n_transitions, obs_dim).astype(np.float32) * 0.5
    actions = np.column_stack([
        rng.randint(0, nv, n_transitions) for nv in act_nvec
    ]).astype(np.float32)

    move_action = actions[:, 0]
    shoot_action = actions[:, 1]

    next_obs = obs.copy()
    angle = move_action * (2 * np.pi / 12)
    speed = 0.05
    next_obs[:, 0] += np.cos(angle) * speed
    next_obs[:, 1] += np.sin(angle) * speed
    next_obs[:, 2] += rng.randn(n_transitions).astype(np.float32) * 0.01

    next_obs[:, 3] = obs[:, 3] * 0.99 + rng.randn(n_transitions).astype(np.float32) * 0.01
    next_obs[:, 4] = obs[:, 4] * 0.99 + rng.randn(n_transitions).astype(np.float32) * 0.01
    next_obs[:, 5] = obs[:, 5] * 0.99 + rng.randn(n_transitions).astype(np.float32) * 0.01

    next_obs[:, 9] = obs[:, 9] - shoot_action * 0.02 + rng.randn(n_transitions).astype(np.float32) * 0.005

    damage = shoot_action * 0.1 * (1 + rng.rand(n_transitions).astype(np.float32) * 0.5)
    rewards = damage - 0.01
    rewards = rewards.astype(np.float32)

    dones = np.zeros(n_transitions, dtype=np.float32)

    buffer = ReplayBuffer(capacity=n_transitions, obs_dim=obs_dim, act_dim=act_dim)
    buffer.add_batch(obs, actions, rewards, next_obs, dones)

    print(f"  Generated {n_transitions:,} synthetic transitions")
    print(f"  Obs dim: {obs_dim}, Act dim: {act_dim}, Act nvec: {act_nvec}")

    return buffer, obs_dim, act_dim, act_nvec


class RandomPolicy:
    """Random policy for evaluation rollouts."""
    def __init__(self, act_nvec):
        self.act_nvec = act_nvec

    def predict(self, obs, deterministic=True):
        n = obs.shape[0] if obs.ndim == 2 else 1
        actions = np.column_stack([
            np.random.randint(0, nv, n) for nv in self.act_nvec
        ]).astype(np.float32)
        return actions, None


def main():
    p = argparse.ArgumentParser(description="World Model Demo")
    p.add_argument("--config", default=None, help="Game config JSON path")
    p.add_argument("--scenario", default="cs_lite")
    p.add_argument("--policy", default=None, help="Trained model .zip to collect expert data from")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (no engine needed)")
    p.add_argument("--load-buffer", default=None, help="Load existing replay buffer .npz")
    p.add_argument("--episodes", type=int, default=50, help="Episodes to collect")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2048)

    p.add_argument("--n-models", type=int, default=5, help="Ensemble size")
    p.add_argument("--hidden", type=int, default=256, help="Hidden layer width")
    p.add_argument("--n-layers", type=int, default=3, help="Hidden layers per member")
    p.add_argument("--train-steps", type=int, default=100, help="Gradient steps for training")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--save-dir", default=None, help="Save model and buffer to this directory")
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("  GHOSTLOBBY WORLD MODEL DEMO")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Step 1: Collect or load data
    # -------------------------------------------------------------------
    print("\n--- Step 1: Data Collection ---")
    act_nvec = [12, 2, 2, 3]

    if args.load_buffer:
        from training.replay_buffer import ReplayBuffer
        buffer = ReplayBuffer.load(args.load_buffer)
        obs_dim = buffer.obs_dim
        act_dim = buffer.act_dim
        print(f"  Loaded buffer from {args.load_buffer}: {len(buffer):,} transitions")
    elif args.synthetic:
        buffer, obs_dim, act_dim, act_nvec = generate_synthetic_data()
    else:
        if args.config is None:
            print("  ERROR: Must provide --config or --synthetic")
            sys.exit(1)
        buffer, obs_dim, act_dim, act_nvec = collect_from_engine(
            args.config, args.scenario, args.episodes,
            args.frame_skip, args.max_steps, policy_path=args.policy,
        )

    print(f"  Buffer: {len(buffer):,} transitions, obs_dim={obs_dim}, act_dim={act_dim}")

    # -------------------------------------------------------------------
    # Step 2: Create and train dynamics ensemble
    # -------------------------------------------------------------------
    print("\n--- Step 2: Train Dynamics Ensemble ---")
    from training.dynamics_model import DynamicsEnsemble
    from training.dynamics_trainer import DynamicsTrainer

    ensemble = DynamicsEnsemble(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_nvec=act_nvec,
        n_models=args.n_models,
        hidden=args.hidden,
        n_layers=args.n_layers,
        lr=args.lr,
    )

    trainer = DynamicsTrainer(
        ensemble, buffer,
        batch_size=args.batch_size,
    )

    print(f"  Ensemble: {args.n_models} members, {args.hidden}x{args.n_layers} hidden")
    print(f"  Training for {args.train_steps} gradient steps...")

    t0 = time.perf_counter()
    n_chunks = max(1, args.train_steps // 10)
    for i in range(n_chunks):
        steps = args.train_steps // n_chunks
        metrics = trainer.train(
            n_steps=steps,
            refit_normalizer=(i == 0),
        )
        elapsed = time.perf_counter() - t0
        print(
            f"  [{(i+1)*steps:4d}/{args.train_steps}] "
            f"loss={metrics['mean_loss']:.4f}, "
            f"val_loss={metrics['val_loss']:.6f}, "
            f"time={elapsed:.1f}s"
        )

    total_time = time.perf_counter() - t0
    print(f"  Training complete in {total_time:.1f}s")

    # -------------------------------------------------------------------
    # Step 3: Evaluate prediction accuracy
    # -------------------------------------------------------------------
    print("\n--- Step 3: Evaluate Prediction Accuracy ---")
    from training.dynamics_eval import evaluate_world_model, print_report

    policy = RandomPolicy(act_nvec)
    results = evaluate_world_model(
        ensemble, buffer, policy,
        n_1step=min(2000, len(buffer)),
        n_rollouts=min(200, len(buffer)),
        horizons=[1, 3, 5, 10, 15, 20],
    )
    print_report(results)

    # -------------------------------------------------------------------
    # Step 4: Generate imagined rollouts
    # -------------------------------------------------------------------
    print("--- Step 4: Imagined Rollouts ---")
    n_demo = 5
    starts = buffer.sample_states(n_demo)
    rollout = ensemble.rollout(starts, policy, horizon=10)

    print(f"  Generated {n_demo} imagined trajectories (horizon=10)")
    for i in range(n_demo):
        start_pos = starts[i, :3]
        end_pos = rollout["observations"][i, 10, :3]
        total_rew = rollout["rewards"][i].sum()
        max_disagree = rollout["disagreements"][i].max()
        print(
            f"  Trajectory {i}: "
            f"start=[{start_pos[0]:.2f},{start_pos[1]:.2f},{start_pos[2]:.2f}] -> "
            f"end=[{end_pos[0]:.2f},{end_pos[1]:.2f},{end_pos[2]:.2f}], "
            f"reward={total_rew:.3f}, max_disagree={max_disagree:.4f}"
        )

    # -------------------------------------------------------------------
    # Step 5: Save artifacts
    # -------------------------------------------------------------------
    if args.save_dir:
        print(f"\n--- Step 5: Save Artifacts ---")
        os.makedirs(args.save_dir, exist_ok=True)
        ensemble.save(os.path.join(args.save_dir, "dynamics_ensemble.pt"))
        buffer.save(os.path.join(args.save_dir, "replay_buffer.npz"))

        import json
        meta = {
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "act_nvec": [int(x) for x in act_nvec],
            "n_models": args.n_models,
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "buffer_size": len(buffer),
            "train_steps": args.train_steps,
            "final_loss": float(metrics["mean_loss"]),
            "final_val_loss": float(metrics["val_loss"]),
            "state_mse": float(results["one_step"]["state_mse"]),
            "reward_mse": float(results["one_step"]["reward_mse"]),
            "median_r2": float(np.median(results["one_step"]["r_squared"])),
        }
        with open(os.path.join(args.save_dir, "world_model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved to {args.save_dir}/")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
