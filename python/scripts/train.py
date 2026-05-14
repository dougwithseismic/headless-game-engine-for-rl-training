#!/usr/bin/env python3
"""Unified training entry point for GhostLobby.

Usage:
    # PPO training
    python scripts/train.py --scenario arena3d --config configs/arena3d/phase1_aim_targets.json --timesteps 3000000

    # BC pre-training
    python scripts/train.py --scenario arena3d --mode bc --demos data/demos/arena3d_aim.npz --config configs/arena3d/phase1_aim_targets.json

    # PPO with BC warm-start and KL anchor
    python scripts/train.py --scenario arena3d --config configs/arena3d/phase1_aim_targets.json \\
        --resume data/bc_models/arena3d_aim.zip --kl-anchor data/bc_models/arena3d_aim_ref.pt \\
        --entropy-schedule 0.01:0.001

    # Full curriculum
    python scripts/train.py --scenario arena3d --mode curriculum --curriculum configs/arena3d/curriculum.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="GhostLobby Training")
    p.add_argument("--scenario", required=True,
                   choices=["cs_lite", "cs-lite", "cs_lite_dummy", "tactical"])
    p.add_argument("--mode", default="ppo", choices=["ppo", "bc", "curriculum"])
    p.add_argument("--config", default=None, help="Game config JSON path")
    p.add_argument("--curriculum", default=None, help="Curriculum YAML path")
    p.add_argument("--demos", default=None, help="Demo .npz path for BC mode")
    p.add_argument("--output", default=None, help="Output path for BC model")
    p.add_argument("--resume", default=None, help="Model .zip to resume from")
    p.add_argument("--kl-anchor", default=None, help="BC reference .pt for KL penalty")
    p.add_argument("--kl-beta-start", type=float, default=0.5)
    p.add_argument("--kl-beta-end", type=float, default=0.0)
    p.add_argument("--kl-anneal-steps", type=int, default=2_000_000)
    p.add_argument("--entropy-schedule", default=None, help="start:end (e.g. 0.01:0.001)")
    p.add_argument("--entropy-schedule-steps", type=int, default=None,
                   help="Timesteps for entropy annealing (default: --timesteps)")
    p.add_argument("--phase", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=4096)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2048)
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--self-play", action="store_true")
    p.add_argument("--swap-interval", type=int, default=500_000)
    p.add_argument("--scripted-warmup", type=int, default=1_000_000)
    p.add_argument("--auto-stop", action="store_true")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--memory", default="none", choices=["none", "lstm", "gru"],
                   help="Recurrent memory architecture (default: none = MLP)")
    p.add_argument("--memory-hidden-size", type=int, default=256)
    p.add_argument("--memory-sequence-length", type=int, default=None,
                   help="Sequence length for recurrent training (default: SB3 default)")
    # Deprecated alias: --lstm maps to --memory lstm
    p.add_argument("--lstm", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--name", default=None)
    p.add_argument("--no-track-behavior", action="store_true",
                   help="Disable behavior tracking during eval")
    p.add_argument("--bc-filter", default="all", choices=["all", "combat", "aim"],
                   help="Filter BC demos: all=every tick, combat=shooting ticks, aim=visible enemy ticks")

    # Dyna world model augmentation
    p.add_argument("--dyna", action="store_true",
                   help="Enable Dyna-style world model augmentation")
    p.add_argument("--dyna-n-models", type=int, default=5,
                   help="Dynamics ensemble size (default: 5)")
    p.add_argument("--dyna-hidden", type=int, default=256,
                   help="Dynamics model hidden width (default: 256)")
    p.add_argument("--dyna-n-layers", type=int, default=3,
                   help="Dynamics model hidden layers (default: 3)")
    p.add_argument("--dyna-buffer", type=int, default=500_000,
                   help="Replay buffer capacity (default: 500000)")
    p.add_argument("--dyna-train-freq", type=int, default=2048,
                   help="Train dynamics every N steps (default: 2048)")
    p.add_argument("--dyna-train-steps", type=int, default=10,
                   help="Gradient steps per dynamics training (default: 10)")
    p.add_argument("--dyna-warmup", type=int, default=5000,
                   help="Minimum buffer size before training dynamics (default: 5000)")
    p.add_argument("--dyna-lr", type=float, default=1e-3,
                   help="Dynamics model learning rate (default: 1e-3)")
    p.add_argument("--dyna-no-shaping", action="store_true",
                   help="Disable reward shaping (only collect data + train model)")
    p.add_argument("--dyna-shaping-coef", type=float, default=0.1,
                   help="Reward shaping coefficient (default: 0.1)")
    p.add_argument("--dyna-shaping-horizon", type=int, default=3,
                   help="Imagination horizon for reward shaping (default: 3)")
    p.add_argument("--dyna-curiosity-coef", type=float, default=0.01,
                   help="Curiosity bonus coefficient (default: 0.01)")
    return p.parse_args()


def run_ppo(args):
    from training.ppo_trainer import PPOTrainer

    entropy_schedule = None
    if args.entropy_schedule:
        parts = args.entropy_schedule.split(":")
        entropy_schedule = (float(parts[0]), float(parts[1]))

    # --lstm is a deprecated alias for --memory lstm
    memory = args.memory
    if args.lstm and memory == "none":
        memory = "lstm"

    trainer = PPOTrainer(
        scenario=args.scenario,
        config_path=args.config,
        name=args.name or args.scenario,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        n_envs=args.n_envs,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        phase=args.phase,
        timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        n_eval_episodes=args.n_eval_episodes,
        resume=args.resume,
        kl_anchor=args.kl_anchor,
        kl_beta_start=args.kl_beta_start,
        kl_beta_end=args.kl_beta_end,
        kl_anneal_steps=args.kl_anneal_steps,
        entropy_schedule=entropy_schedule,
        entropy_schedule_steps=args.entropy_schedule_steps,
        self_play=args.self_play,
        swap_interval=args.swap_interval,
        scripted_warmup=args.scripted_warmup,
        auto_stop=args.auto_stop,
        patience=args.patience,
        memory=memory,
        memory_hidden_size=args.memory_hidden_size,
        memory_sequence_length=args.memory_sequence_length,
        track_behavior=not args.no_track_behavior,
        dyna=args.dyna,
        dyna_n_models=args.dyna_n_models,
        dyna_hidden=args.dyna_hidden,
        dyna_n_layers=args.dyna_n_layers,
        dyna_buffer_capacity=args.dyna_buffer,
        dyna_train_freq=args.dyna_train_freq,
        dyna_train_steps=args.dyna_train_steps,
        dyna_warmup=args.dyna_warmup,
        dyna_lr=args.dyna_lr,
        dyna_shaping=not args.dyna_no_shaping,
        dyna_shaping_coef=args.dyna_shaping_coef,
        dyna_shaping_horizon=args.dyna_shaping_horizon,
        dyna_curiosity_coef=args.dyna_curiosity_coef,
    )
    return trainer.train()


def run_bc(args):
    from training.bc_pretrain import BCTrainer
    from training.ppo_trainer import _import_gym_class

    demos_path = args.demos
    if not demos_path:
        if not args.config:
            print("Error: --demos or --config required for BC mode")
            sys.exit(1)
        # Auto-collect demos from scripted AI
        from training.bc_collector import collect_demonstrations
        demos_path = f"data/demos/{args.scenario}_auto.npz"
        os.makedirs(os.path.dirname(demos_path), exist_ok=True)
        print(f"No --demos provided, collecting from scripted AI (filter={args.bc_filter})...")
        collect_demonstrations(
            config_path=args.config,
            scenario=args.scenario,
            num_episodes=500,
            output_path=demos_path,
            phase=args.phase,
            filter_mode=args.bc_filter,
        )

    import numpy as np
    data = np.load(demos_path)
    obs_dim = data["observations"].shape[1]

    gym_class = _import_gym_class(args.scenario)
    config = args.config
    if not config:
        print("Error: --config required for BC mode (needed for branch sizes)")
        sys.exit(1)
    tmp_env = gym_class(config_path=config, scenario=args.scenario, phase=args.phase)
    branch_sizes = list(tmp_env.action_space.nvec)
    tmp_env.close()

    trainer = BCTrainer(obs_dim=obs_dim, branch_sizes=branch_sizes)
    stats = trainer.train(demos_path, epochs=50)

    output = args.output or f"data/bc_models/{args.scenario}"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Save reference .pt for KL anchor
    trainer.save_reference(f"{output}_ref.pt")

    # Save SB3-compatible model
    env = gym_class(config_path=config, scenario=args.scenario, phase=args.phase)
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    trainer.save_as_sb3(output, vec_env)
    vec_env.close()

    print(f"BC model: {output}.zip")
    print(f"BC reference: {output}_ref.pt")


def run_curriculum(args):
    from training.curriculum import CurriculumRunner

    if not args.curriculum:
        print("Error: --curriculum required for curriculum mode")
        sys.exit(1)

    runner = CurriculumRunner(args.curriculum)
    final = runner.run(resume_model=args.resume)
    print(f"Final model: {final}")


def main():
    args = parse_args()

    if args.mode == "ppo":
        if not args.config:
            print("Error: --config required for PPO mode")
            sys.exit(1)
        run_ppo(args)
    elif args.mode == "bc":
        run_bc(args)
    elif args.mode == "curriculum":
        run_curriculum(args)


if __name__ == "__main__":
    main()
