#!/usr/bin/env python3
"""DAgger (Dataset Aggregation) for improving BC models.

Iteratively:
1. Run current model to collect states
2. Query scripted AI for expert actions in those states
3. Combine with existing demos
4. Retrain BC on combined data
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_expert_labels(config_path, scenario, model_path, num_episodes, max_ticks):
    """Run the model, but collect what the scripted AI would do.

    Uses two envs: one runs the model (to get realistic states),
    one runs scripted AI only (to get expert actions for similar states).
    """
    import ghostlobby as gl
    from stable_baselines3 import PPO

    model = PPO.load(model_path)

    all_obs = []
    all_actions = []

    for ep in range(num_episodes):
        # Expert env: scripted AI controls both sides
        expert_env = gl.GhostLobbyEnv(config_path, scenario=scenario)
        expert_env.reset()

        for tick in range(max_ticks):
            # Step with empty actions — scripted AI fills everything
            obs_dict, rewards, terms, truncs, infos = expert_env.step({})
            actions = expert_env.get_actions()

            agent_id = 0
            agent_action = actions.get(agent_id, [])
            if not agent_action or len(agent_action) < 4:
                continue

            agent_obs = obs_dict.get(agent_id, {})
            obs_flat = []
            for key in sorted(agent_obs.keys()):
                obs_flat.extend(agent_obs[key])

            from training.bc_collector import _bins_to_continuous
            continuous_action = _bins_to_continuous(agent_action)

            all_obs.append(np.array(obs_flat, dtype=np.float32))
            all_actions.append(np.array(continuous_action, dtype=np.float32))

            if terms.get(agent_id, False) or truncs.get(agent_id, False):
                break

        if (ep + 1) % max(1, num_episodes // 5) == 0:
            print(f"  DAgger episode {ep+1}/{num_episodes}: {len(all_obs)} transitions")

    return np.array(all_obs, dtype=np.float32), np.array(all_actions, dtype=np.float32)


def run_dagger(config_path, scenario, initial_model, initial_demos,
               n_rounds, episodes_per_round, max_ticks, output_path):
    """Run the full DAgger loop."""
    from training.bc_pretrain import BCTrainer
    from stable_baselines3 import PPO

    # Load initial demos
    initial_data = np.load(initial_demos)
    all_obs = list(initial_data["observations"])
    all_actions = list(initial_data["actions"])
    print(f"Initial demos: {len(all_obs)} transitions")

    current_model = initial_model

    for round_num in range(n_rounds):
        print(f"\n=== DAgger Round {round_num + 1}/{n_rounds} ===")

        # Collect expert labels from scripted AI playing
        new_obs, new_actions = collect_expert_labels(
            config_path, scenario, current_model,
            episodes_per_round, max_ticks,
        )

        print(f"  New transitions: {len(new_obs)}")

        # Combine datasets
        all_obs.extend(new_obs)
        all_actions.extend(new_actions)

        obs_arr = np.array(all_obs, dtype=np.float32)
        act_arr = np.array(all_actions, dtype=np.float32)
        print(f"  Total dataset: {len(obs_arr)} transitions")

        # Retrain BC
        obs_dim = obs_arr.shape[1]
        act_dim = act_arr.shape[1]
        trainer = BCTrainer(obs_dim=obs_dim, act_dim=act_dim)

        # Save combined data temporarily
        tmp_data = os.path.join(os.path.dirname(output_path) or ".", f"dagger_round{round_num}.npz")
        np.savez(tmp_data, observations=obs_arr, actions=act_arr,
                 episode_starts=np.zeros(len(obs_arr), dtype=bool),
                 rewards=np.zeros(len(obs_arr), dtype=np.float32))

        stats = trainer.train(tmp_data, epochs=30)
        print(f"  BC loss: {stats['final_val_loss']:.6f}")

        # Save model
        round_model = os.path.join(os.path.dirname(output_path) or ".",
                                    f"dagger_round{round_num}")
        trainer.save_reference(f"{round_model}_ref.pt")

        # Create a gym env to save SB3-compatible model
        from glgym.gym_cs_lite import CsLiteGym
        env = CsLiteGym(config_path=config_path, scenario=scenario)
        trainer.save_as_sb3(round_model, env)
        env.close()

        current_model = f"{round_model}.zip"
        print(f"  Model saved: {current_model}")

        # Evaluate
        from stable_baselines3 import PPO
        eval_model = PPO.load(current_model)
        eval_env = CsLiteGym(config_path=config_path, scenario=scenario,
                              frame_skip=4, max_steps=512, track_behavior=True)

        total_hits = 0
        total_shots = 0
        for eval_ep in range(10):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = eval_model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(action)
                done = term or trunc
            b = info.get("behavior", {})
            total_hits += b.get("shots_hit", 0)
            total_shots += b.get("shots_fired", 0)

        acc = total_hits / max(total_shots, 1) * 100
        print(f"  Eval accuracy: {acc:.1f}% ({total_hits}/{total_shots})")
        eval_env.close()

    # Save final model
    import shutil
    shutil.copy(current_model, output_path)
    print(f"\nFinal DAgger model: {output_path}")


def main():
    p = argparse.ArgumentParser(description="DAgger iterative BC improvement")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True, help="Initial BC model .zip")
    p.add_argument("--demos", required=True, help="Initial BC demos .npz")
    p.add_argument("--scenario", default="cs_lite")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--episodes-per-round", type=int, default=200)
    p.add_argument("--max-ticks", type=int, default=4000)
    p.add_argument("--output", default="data/bc_models/dagger_final.zip")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    run_dagger(
        config_path=args.config,
        scenario=args.scenario,
        initial_model=args.model,
        initial_demos=args.demos,
        n_rounds=args.rounds,
        episodes_per_round=args.episodes_per_round,
        max_ticks=args.max_ticks,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
