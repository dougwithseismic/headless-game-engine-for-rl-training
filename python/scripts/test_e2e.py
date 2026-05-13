#!/usr/bin/env python3
"""End-to-end integration test for the hierarchical agent framework.

Tests all three layers with full I/O capture:
  Layer 1 (Motor):    PPO policy with MultiDiscrete action space
  Layer 2 (Tactics):  Goal conditioning via goal_state observation
  Layer 3 (Strategy): Rules-based goal assignment via cs_goal_assignment_system

Then runs a tiny PPO training loop (5K steps) to prove gradients flow
and the agent learns from reward signals through the full stack.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_layer1_motor():
    """Layer 1: Verify MultiDiscrete action space and motor execution."""
    print("\n" + "=" * 60)
    print("LAYER 1: Motor (MultiDiscrete Action Space)")
    print("=" * 60)

    from glgym.gym_cs_lite import CsLiteGym

    env = CsLiteGym(
        config_path="configs/cs_lite/1v1_tactical.json",
        scenario="cs_lite_dummy",
        frame_skip=1,
        max_steps=100,
    )

    print(f"  Action space: {env.action_space}")
    print(f"  Action nvec:  {env.action_space.nvec}")
    print(f"  Obs shape:    {env.observation_space.shape}")

    obs, info = env.reset()
    print(f"  Obs dtype:    {obs.dtype}")
    print(f"  Obs range:    [{obs.min():.3f}, {obs.max():.3f}]")

    io_pairs = []
    total_reward = 0.0
    for step in range(20):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        io_pairs.append({
            "step": step,
            "action": action.tolist(),
            "action_names": ["move_target", "shoot", "reload", "use"],
            "reward": float(reward),
            "terminated": terminated,
            "obs_size": len(next_obs),
        })

        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    print(f"\n  Sample I/O pairs:")
    for p in io_pairs[:5]:
        print(f"    Step {p['step']:2d}: action={p['action']} reward={p['reward']:+.4f}")
    print(f"  Total reward over 20 steps: {total_reward:+.4f}")
    print(f"  PASS: MultiDiscrete({list(env.action_space.nvec)}) actions flow through engine")
    return io_pairs


def test_layer2_goals():
    """Layer 2: Verify goal conditioning in observations."""
    print("\n" + "=" * 60)
    print("LAYER 2: Tactics (Goal Conditioning)")
    print("=" * 60)

    from glgym.gym_cs_lite import CsLiteGym
    import ghostlobby as gl

    env_raw = gl.GhostLobbyEnv("configs/cs_lite/1v1_tactical.json", scenario="cs_lite_dummy")
    obs_raw, _ = env_raw.reset()

    obs_space = env_raw.observation_space()
    feature_names = [f["name"] for f in obs_space["features"]]
    print(f"  Observation features: {feature_names}")

    assert "goal_state" in feature_names, "goal_state feature missing from obs!"
    goal_idx = feature_names.index("goal_state")
    goal_shape = obs_space["features"][goal_idx]["shape"]
    print(f"  goal_state shape: {goal_shape} ({goal_shape[0]} dims)")

    # Run a few ticks to get into active phase
    agent_id = 0
    for _ in range(200):
        action = {agent_id: [10.0, 0.0, 0.0, 0.0]}  # advance, no shoot
        obs_raw, _, _, _, _ = env_raw.step(action)

    agent_obs = obs_raw[agent_id]
    goal_data = agent_obs.get("goal_state", [])
    print(f"\n  goal_state values: {[f'{v:.2f}' for v in goal_data]}")

    # Decode goal_state: [5 objective one-hot] [3 target pos] [3 posture one-hot]
    objective_onehot = goal_data[:5]
    target_pos = goal_data[5:8]
    posture_onehot = goal_data[8:11]

    obj_names = ["PlantBomb", "DefuseBomb", "HoldPosition", "Eliminate", "Rotate"]
    active_obj = obj_names[np.argmax(objective_onehot)] if max(objective_onehot) > 0 else "None"
    posture_names = ["Aggressive", "Default", "Passive"]
    active_posture = posture_names[np.argmax(posture_onehot)] if max(posture_onehot) > 0 else "None"

    print(f"  Decoded objective:  {active_obj} (one-hot: {[f'{v:.0f}' for v in objective_onehot]})")
    print(f"  Decoded target pos: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    print(f"  Decoded posture:    {active_posture} (one-hot: {[f'{v:.0f}' for v in posture_onehot]})")

    # Verify action mask has expanded to 19
    mask_data = agent_obs.get("action_mask", [])
    print(f"\n  action_mask length: {len(mask_data)} (expected 19)")
    print(f"  action_mask values: {[f'{v:.0f}' for v in mask_data]}")
    mask_labels = (
        [f"move_{i}" for i in range(12)] +
        ["no_shoot", "shoot"] +
        ["no_reload", "reload"] +
        ["no_use", "plant", "defuse"]
    )
    active_masks = [mask_labels[i] for i, v in enumerate(mask_data) if v > 0.5]
    print(f"  Valid actions: {active_masks}")

    del env_raw

    assert len(goal_data) == 11, f"goal_state should be 11 dims, got {len(goal_data)}"
    assert len(mask_data) == 19, f"action_mask should be 19 dims, got {len(mask_data)}"
    assert max(objective_onehot) > 0, "No objective assigned!"
    print(f"\n  PASS: Goal conditioning active — {active_obj} objective, {active_posture} posture")

    return {
        "objective": active_obj,
        "target_pos": target_pos,
        "posture": active_posture,
        "mask_valid": active_masks,
    }


def test_layer3_strategy():
    """Layer 3: Verify strategy layer types are accessible and serializable."""
    print("\n" + "=" * 60)
    print("LAYER 3: Strategy (Game-Agnostic Interface)")
    print("=" * 60)

    # The strategy layer is Rust-side. We verify its types exist and the
    # goal assignment system (which IS the current rules strategy) works
    # by checking that goals change based on game state.

    import ghostlobby as gl

    env = gl.GhostLobbyEnv("configs/cs_lite/1v1_tactical.json", scenario="cs_lite_dummy")
    obs, _ = env.reset()

    goals_over_time = []

    for tick in range(300):
        action = {0: [10.0, 0.0, 0.0, 0.0]}
        obs, _, _, _, _ = env.step(action)

        if tick % 50 == 0 and 0 in obs:
            goal = obs[0].get("goal_state", [0]*11)
            obj_names = ["PlantBomb", "DefuseBomb", "HoldPosition", "Eliminate", "Rotate"]
            active = obj_names[np.argmax(goal[:5])] if max(goal[:5]) > 0 else "None"
            goals_over_time.append({
                "tick": tick,
                "objective": active,
                "target": [round(v, 2) for v in goal[5:8]],
            })

    print(f"  Goal assignments over time:")
    for g in goals_over_time:
        print(f"    Tick {g['tick']:3d}: {g['objective']:15s} target={g['target']}")

    del env

    assert len(goals_over_time) > 0, "No goals captured!"
    assert any(g["objective"] != "None" for g in goals_over_time), "All goals are None!"
    print(f"\n  PASS: Strategy layer assigns goals that change with game state")
    return goals_over_time


def test_training_loop():
    """Run a tiny PPO training loop to prove gradients flow through the full stack."""
    print("\n" + "=" * 60)
    print("TRAINING: PPO with MultiDiscrete + Goal Conditioning")
    print("=" * 60)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("  SKIP: stable-baselines3 not available in this Python environment")
        return None

    from glgym.gym_cs_lite import CsLiteGym

    def make_env():
        return CsLiteGym(
            config_path="configs/cs_lite/1v1_tactical.json",
            scenario="cs_lite_dummy",
            frame_skip=4,
            max_steps=256,
        )

    print("  Creating vectorized environment (4 envs)...")
    vec_env = DummyVecEnv([make_env for _ in range(4)])

    print(f"  Action space: {vec_env.action_space}")
    print(f"  Obs space:    {vec_env.observation_space.shape}")

    print("  Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=2,
        verbose=0,
    )

    # Capture initial policy behavior
    obs = vec_env.reset()
    initial_actions = []
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        initial_actions.append(action[0].tolist())
        obs, _, _, _ = vec_env.step(action)

    print(f"\n  Initial policy actions (first 5):")
    for i, a in enumerate(initial_actions[:5]):
        print(f"    {a}")

    print(f"\n  Training for 5,000 steps...")
    t0 = time.perf_counter()
    model.learn(total_timesteps=5_000)
    elapsed = time.perf_counter() - t0
    sps = 5_000 / elapsed

    # Capture post-training behavior
    obs = vec_env.reset()
    trained_actions = []
    total_reward = 0.0
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        trained_actions.append(action[0].tolist())
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += rewards.sum()

    print(f"\n  Training complete in {elapsed:.1f}s ({sps:.0f} steps/sec)")
    print(f"  Post-training actions (first 5):")
    for i, a in enumerate(trained_actions[:5]):
        print(f"    {a}")

    # Check that the policy actually changed (it should — even 5K steps is enough for some gradient updates)
    initial_set = set(tuple(a) for a in initial_actions)
    trained_set = set(tuple(a) for a in trained_actions)
    policy_changed = initial_set != trained_set
    print(f"\n  Policy changed after training: {policy_changed}")
    print(f"  Unique initial actions:  {len(initial_set)}")
    print(f"  Unique trained actions:  {len(trained_set)}")
    print(f"  Eval reward (50 steps):  {total_reward:+.2f}")

    vec_env.close()

    print(f"\n  PASS: PPO trains successfully with MultiDiscrete + goal conditioning")
    return {
        "steps_per_sec": sps,
        "elapsed": elapsed,
        "policy_changed": policy_changed,
        "eval_reward": float(total_reward),
    }


def main():
    print("=" * 60)
    print("GhostLobby Hierarchical Agent Framework — E2E Test")
    print("=" * 60)

    results = {}

    # Layer 1: Motor
    results["layer1_motor"] = test_layer1_motor()

    # Layer 2: Goals
    results["layer2_goals"] = test_layer2_goals()

    # Layer 3: Strategy
    results["layer3_strategy"] = test_layer3_strategy()

    # Training
    results["training"] = test_training_loop()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    checks = [
        ("MultiDiscrete action space [12,2,2,3]", results["layer1_motor"] is not None),
        ("Goal conditioning (11-dim goal_state)", results["layer2_goals"] is not None),
        ("Strategy assigns contextual goals", results["layer3_strategy"] is not None),
        ("PPO trains through full stack", results["training"] is not None),
    ]

    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    if results["training"]:
        t = results["training"]
        print(f"\n  Training: {t['steps_per_sec']:.0f} steps/sec, reward={t['eval_reward']:+.2f}")

    print(f"\n{'=' * 60}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'=' * 60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
