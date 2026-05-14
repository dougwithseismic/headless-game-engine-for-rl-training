"""Train an RL agent to play Pokemon Gold.

Usage:
    # Quick test (random exploration, 50K steps)
    python train_gold.py --steps 50000

    # Full training with exploration reward
    python train_gold.py --reward exploration --steps 1000000 --envs 4

    # With progress reward (badges + levels + exploration)
    python train_gold.py --reward progress --steps 2000000 --envs 8

    # From save state (skip title screens)
    python train_gold.py --state data/pokemon_gold_ready.state --steps 500000

    # With temporal features + frame stacking (recommended)
    python train_gold.py --temporal --frame-stack 4 --steps 1000000

    # With anti-loop penalties
    python train_gold.py --temporal --anti-loop --steps 1000000

    # With screen observations (slower, but richer)
    python train_gold.py --screen --steps 500000

    # High gamma for long-horizon Pokemon play
    python train_gold.py --reward progress --gamma 0.998 --steps 2000000

    # Evaluate a trained model
    python train_gold.py --eval runs/gold_latest/best_model.zip --episodes 5
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

ROM = os.environ.get("POKEMON_ROM", "data/pokemon_gold.gbc")


def make_env(
    rom: str,
    state: str | None,
    reward_type: str,
    max_steps: int,
    include_screen: bool,
    ticks_per_step: int,
    use_temporal: bool = False,
    use_anti_loop: bool = False,
    headed: bool = False,
    rank: int = 0,
):
    """Factory for SubprocVecEnv — must be a top-level function."""
    def _init():
        from bridges.profiles.pokemon_gold import (
            make_pokemon_gold_bridge,
            ExplorationReward,
            ProgressReward,
            BattleReward,
        )
        from bridges.rewards.milestone import MilestoneReward
        from glgym.gym_external import ExternalGameGym

        show_window = headed and rank == 0
        bridge = make_pokemon_gold_bridge(
            rom_path=rom,
            include_screen=include_screen,
            ticks_per_step=ticks_per_step,
            headless=not show_window,
            speed=3 if show_window else 0,
            save_state_path=state,
        )

        idx = bridge.feature_index
        if reward_type == "exploration":
            reward_fn = ExplorationReward(idx)
        elif reward_type == "progress":
            reward_fn = ProgressReward(idx)
        elif reward_type == "battle":
            reward_fn = BattleReward(idx)
        elif reward_type == "milestone":
            reward_fn = MilestoneReward(idx)
        else:
            reward_fn = None

        env = ExternalGameGym(
            bridge=bridge,
            reward_fn=reward_fn,
            max_steps=max_steps,
        )

        if use_temporal:
            from bridges.wrappers.temporal import TemporalObsWrapper
            env = TemporalObsWrapper(env, feature_index=idx)

        if use_anti_loop:
            from bridges.wrappers.anti_loop import AntiLoopWrapper
            env = AntiLoopWrapper(
                env,
                spam_threshold=5,
                spam_penalty=-0.02,
                heavy_spam_threshold=10,
                heavy_spam_penalty=-0.05,
                cycle_window=8,
                cycle_penalty=-0.03,
                idle_threshold=15,
                idle_penalty=-0.05,
            )

        from bridges.wrappers.pokemon_telemetry import PokemonTelemetryWrapper
        env = PokemonTelemetryWrapper(env, feature_index=idx)

        if include_screen:
            from bridges.wrappers.multi_input import MultiInputWrapper
            from bridges.profiles.pokemon_gold import POKEMON_GOLD_RAM
            n_ram = len(POKEMON_GOLD_RAM)
            if use_temporal:
                from bridges.wrappers.temporal import N_DELTA_FEATURES
                n_ram += N_DELTA_FEATURES
            env = MultiInputWrapper(env, n_ram=n_ram)

        return env
    return _init


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
    )

    run_name = f"gold_{args.reward}_{int(time.time())}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    env_fns = [
        make_env(args.rom, args.state, args.reward, args.max_steps,
                 args.screen, args.ticks, args.temporal, args.anti_loop,
                 args.headed, rank=i)
        for i in range(args.envs)
    ]

    if args.envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    eval_env = DummyVecEnv([
        make_env(args.rom, args.state, args.reward, args.max_steps,
                 args.screen, args.ticks, args.temporal, args.anti_loop,
                 headed=False, rank=0)
    ])

    if args.frame_stack > 1 and not args.screen:
        vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)
        eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)

    if args.screen:
        policy_type = "MultiInputPolicy"
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )
    else:
        policy_type = "MlpPolicy"
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )

    model = PPO(
        policy_type,
        vec_env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
    )

    from bridges.wrappers.gold_callbacks import GoldTelemetryCallback
    from bridges.wrappers.pokemon_eval import PokemonEvalCallback

    callbacks = [
        GoldTelemetryCallback(verbose=0),
        PokemonEvalCallback(
            eval_env,
            eval_freq=max(args.steps // (10 * args.envs), 5000),
            n_eval_episodes=5,
            log_path=str(run_dir / "honest_eval"),
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(args.steps // (20 * args.envs), 1000),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="gold",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "eval_logs"),
            eval_freq=max(args.steps // (10 * args.envs), 5000),
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    features = []
    if args.temporal:
        features.append("temporal-deltas")
    if args.anti_loop:
        features.append("anti-loop")
    if args.frame_stack > 1:
        features.append(f"frame-stack-{args.frame_stack}")

    print(f"Training: {args.steps} steps, {args.envs} envs, reward={args.reward}")
    print(f"Features: {', '.join(features) or 'none'}")
    print(f"Gamma: {args.gamma}, LR: {args.lr}, Ent: {args.ent_coef}")
    print(f"Obs space: {vec_env.observation_space.shape}")
    print(f"Act space: {vec_env.action_space}")

    t0 = time.monotonic()
    model.learn(total_timesteps=args.steps, callback=callbacks)
    elapsed = time.monotonic() - t0

    model.save(str(run_dir / "final_model"))
    print(f"\nDone in {elapsed:.0f}s ({args.steps/elapsed:.0f} steps/sec)")
    print(f"Model saved to {run_dir}")

    vec_env.close()
    eval_env.close()


def evaluate(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    env_fn = make_env(
        args.rom, args.state, args.reward, args.max_steps,
        args.screen, args.ticks, args.temporal, args.anti_loop,
        headed=args.headed, rank=0,
    )
    vec_env = DummyVecEnv([env_fn])
    if args.frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)

    model = PPO.load(args.eval, env=vec_env)
    print(f"Loaded model: {args.eval}")

    rewards = []
    for ep in range(args.episodes):
        obs = vec_env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            steps += 1
        rewards.append(total_reward)
        print(f"  Episode {ep+1}: reward={total_reward:.1f}, steps={steps}")

    print(f"\nMean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL on Pokemon Gold")
    parser.add_argument("--rom", default=ROM)
    parser.add_argument("--state", default=None, help="Save state file")
    parser.add_argument("--reward", default="exploration",
                        choices=["exploration", "progress", "battle", "milestone", "none"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--max-steps", type=int, default=2048, help="Steps per episode")
    parser.add_argument("--envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--screen", action="store_true", help="Include screen pixels")
    parser.add_argument("--ticks", type=int, default=24, help="Emulator frames per step")
    parser.add_argument("--temporal", action="store_true",
                        help="Add delta features (velocity, HP change, stuck counter)")
    parser.add_argument("--anti-loop", action="store_true",
                        help="Penalize repetitive action patterns")
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Number of frames to stack (1 = no stacking)")
    parser.add_argument("--headed", action="store_true",
                        help="Show PyBoy window for env 0 (watch the agent play). "
                             "Auto-sets max-steps to 8192 unless overridden.")
    parser.add_argument("--gamma", type=float, default=0.998,
                        help="Discount factor (0.998 for long-horizon Pokemon)")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--eval", default=None, help="Path to model.zip to evaluate")
    parser.add_argument("--episodes", type=int, default=5, help="Eval episodes")
    args = parser.parse_args()

    if args.headed and args.max_steps == 2048:
        args.max_steps = 8192

    if args.eval:
        evaluate(args)
    else:
        train(args)
