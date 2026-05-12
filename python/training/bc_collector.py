"""Behavioral Cloning demo collector for GhostLobby.

Collects (observation, optimal_action) pairs by running an expert policy
or the Rust-side scripted AI (preferred). The native collector works with
any scenario that has scripted_ai enabled in the engine.

Typical usage::

    from training.bc_collector import collect_demonstrations

    collect_demonstrations(
        config_path="configs/cs_lite/cs_lite.json",
        scenario="cs_lite",
        num_episodes=500,
        output_path="data/cs_lite_demos.npz",
    )

Output format (.npz):
    observations : float32 array [N, obs_size]
    actions      : float32 array [N, action_size]
    episode_starts : bool array  [N]
    rewards      : float32 array [N]
"""

import os
from typing import Any, Callable

import numpy as np


def collect_demonstrations(
    config_path: str,
    scenario: str,
    num_episodes: int,
    output_path: str,
    phase: int | None = None,
    max_steps_per_episode: int = 2048,
    frame_skip: int = 4,
    expert_policy: Any | None = None,
    _env_factory: Callable | None = None,
    filter_mode: str = "all",
) -> dict:
    """Collect (observation, optimal_action) pairs using an expert policy.

    Runs the gym environment for ``num_episodes`` episodes. At each step,
    the expert policy reads the current observation and produces the optimal
    action. Both the observation and the action are recorded.

    Parameters
    ----------
    config_path : str
        Path to a GhostLobby JSON config file.
    scenario : str
        Scenario name (e.g. "cs_lite", "tactical").
    num_episodes : int
        Number of episodes to collect.
    output_path : str
        Path to save the output .npz file.
    phase : int or None
        Curriculum phase for the gym wrapper.
    max_steps_per_episode : int
        Maximum steps per episode before truncation.
    frame_skip : int
        Engine ticks per gym step.
    expert_policy : object or None
        An object with a `predict(obs)` method returning an action array.
        If None, delegates to ``collect_demonstrations_native()`` which
        uses the Rust scripted AI directly.
    _env_factory : callable or None
        Internal testing hook. If provided, called with keyword arguments
        to create the environment instead of importing from glgym.

    Returns
    -------
    dict
        Summary statistics: num_transitions, num_episodes, mean_reward.
    """
    if expert_policy is None:
        # Prefer native collector (uses Rust scripted AI directly)
        return collect_demonstrations_native(
            config_path=config_path,
            scenario=scenario,
            num_episodes=num_episodes,
            output_path=output_path,
            max_ticks_per_episode=max_steps_per_episode * frame_skip,
            filter_mode=filter_mode,
        )

    # Create the environment
    if _env_factory is not None:
        env = _env_factory(
            config_path=config_path,
            scenario=scenario,
            phase=phase,
            max_steps=max_steps_per_episode,
            frame_skip=frame_skip,
        )
    else:
        from glgym.gym_tactical import TacticalGym
        env = TacticalGym(
            config_path=config_path,
            scenario=scenario,
            phase=phase,
            max_steps=max_steps_per_episode,
            frame_skip=frame_skip,
        )

    all_obs = []
    all_actions = []
    all_episode_starts = []
    all_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done:
            action = expert_policy.predict(obs)

            all_obs.append(obs.copy())
            all_actions.append(action.copy())
            all_episode_starts.append(step == 0)

            obs, reward, terminated, truncated, info = env.step(action)
            all_rewards.append(reward)

            done = terminated or truncated
            step += 1

        if (ep + 1) % max(1, num_episodes // 10) == 0:
            ep_reward = sum(all_rewards[-step:])
            print(
                f"  Episode {ep + 1}/{num_episodes}: "
                f"steps={step}, reward={ep_reward:.2f}"
            )

    env.close()

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_actions, dtype=np.float32)
    starts_arr = np.array(all_episode_starts, dtype=bool)
    rewards_arr = np.array(all_rewards, dtype=np.float32)

    np.savez(
        output_path,
        observations=obs_arr,
        actions=act_arr,
        episode_starts=starts_arr,
        rewards=rewards_arr,
    )

    total = len(all_obs)
    mean_reward = float(rewards_arr.mean()) if total > 0 else 0.0

    print(
        f"\nSaved {total} transitions from {num_episodes} episodes "
        f"to {output_path}"
    )
    print(f"  Obs shape: {obs_arr.shape}, Actions shape: {act_arr.shape}")
    print(f"  Mean reward: {mean_reward:.4f}")

    return {
        "num_transitions": total,
        "num_episodes": num_episodes,
        "mean_reward": mean_reward,
    }


def _bins_to_continuous(action: list[float]) -> list[float]:
    """Convert raw engine action values to continuous [-1, 1] for the gym."""
    if len(action) < 2:
        return action
    return [
        max(-1.0, min(1.0, action[0] / 5.5 - 1.0)),  # move: bin -> continuous
        1.0 if action[1] > 0.5 else -1.0,              # shoot: binary
    ]


def collect_demonstrations_native(
    config_path: str,
    scenario: str,
    num_episodes: int,
    output_path: str,
    max_ticks_per_episode: int = 2048,
    filter_mode: str = "all",
) -> dict:
    """Collect demonstrations using the Rust scripted AI directly via get_actions().

    This is the preferred method — it works for ANY scenario without writing
    a Python expert policy. The Rust engine's scripted AI computes optimal
    actions, and get_actions() exposes them to Python.

    Parameters
    ----------
    config_path : str
        Path to a GhostLobby JSON config file.
    scenario : str
        Scenario name (e.g. "cs_lite", "tactical").
    num_episodes : int
        Number of episodes to collect.
    output_path : str
        Path to save the output .npz file.
    max_ticks_per_episode : int
        Maximum engine ticks per episode.

    Returns
    -------
    dict
        Summary statistics: num_transitions, num_episodes, mean_reward, mean_ep_len.
    """
    import ghostlobby as gl

    all_obs = []
    all_actions = []
    all_episode_starts = []
    all_rewards = []
    ep_rewards = []

    for ep in range(num_episodes):
        env = gl.GhostLobbyEnv(config_path, scenario=scenario)
        obs_dict, _ = env.reset()

        agent_ids = env.agents()
        agent_id = agent_ids[0] if agent_ids else 0

        ep_reward = 0.0
        step = 0

        for tick in range(max_ticks_per_episode):
            # Step with empty actions — Rust scripted AI fills the buffer
            obs_dict, rewards, terms, truncs, infos = env.step({})

            # Read what the scripted AI actually did
            actions = env.get_actions()

            agent_action = actions.get(agent_id, [])
            if not agent_action:
                continue

            # Filter based on mode
            if filter_mode == "combat":
                # Only collect when agent is shooting AND action has shoot > 0.5
                if len(agent_action) < 4 or agent_action[3] < 0.5:
                    continue
            elif filter_mode == "aim":
                # Only collect when there's likely a visible enemy
                # Check if enemy_state in obs has non-zero values
                agent_obs = obs_dict.get(agent_id, {})
                enemy_state = agent_obs.get("enemy_state", [])
                if all(v == 0.0 for v in enemy_state):
                    continue

            # Flatten the observation
            agent_obs = obs_dict.get(agent_id, {})
            obs_flat = []
            for key in sorted(agent_obs.keys()):
                obs_flat.extend(agent_obs[key])
            obs_arr = np.array(obs_flat, dtype=np.float32)

            all_obs.append(obs_arr)
            continuous_action = _bins_to_continuous(agent_action)
            all_actions.append(np.array(continuous_action, dtype=np.float32))
            all_episode_starts.append(step == 0)

            reward = rewards.get(agent_id, 0.0)
            all_rewards.append(reward)
            ep_reward += reward
            step += 1

            if terms.get(agent_id, False) or truncs.get(agent_id, False):
                break

        ep_rewards.append(ep_reward)

        if (ep + 1) % max(1, num_episodes // 10) == 0:
            print(
                f"  Episode {ep + 1}/{num_episodes}: "
                f"ticks={step}, reward={ep_reward:.2f}"
            )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_actions, dtype=np.float32)
    starts_arr = np.array(all_episode_starts, dtype=bool)
    rewards_arr = np.array(all_rewards, dtype=np.float32)

    np.savez(
        output_path,
        observations=obs_arr,
        actions=act_arr,
        episode_starts=starts_arr,
        rewards=rewards_arr,
    )

    mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    mean_len = float(np.mean([len(r) for r in [all_rewards]])) if all_rewards else 0.0

    print(f"\nSaved {len(all_obs)} transitions from {num_episodes} episodes to {output_path}")
    print(f"  Obs shape: {obs_arr.shape}, Actions shape: {act_arr.shape}")
    print(f"  Mean episode reward: {mean_reward:.2f}")

    return {
        "num_transitions": len(all_obs),
        "num_episodes": num_episodes,
        "mean_reward": mean_reward,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect BC demonstrations from an expert policy."
    )
    parser.add_argument(
        "--config",
        default="configs/cs_lite/cs_lite.json",
        help="Path to the GhostLobby JSON config file.",
    )
    parser.add_argument(
        "--scenario",
        default="cs_lite",
        help="Scenario name.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes to collect.",
    )
    parser.add_argument(
        "--output",
        default="data/cs_lite_demos.npz",
        help="Output .npz file path.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help="Curriculum phase (1-5). None = no masking.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2048,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Engine ticks per gym step.",
    )
    parser.add_argument(
        "--filter",
        default="all",
        choices=["all", "combat", "aim"],
        help="Filter demos: all=every tick, combat=shooting ticks, aim=visible enemy ticks.",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    collect_demonstrations(
        config_path=args.config,
        scenario=args.scenario,
        num_episodes=args.episodes,
        output_path=args.output,
        phase=args.phase,
        max_steps_per_episode=args.max_steps,
        frame_skip=args.frame_skip,
        filter_mode=args.filter,
    )
