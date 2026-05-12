#!/usr/bin/env python3
"""Collect BC demonstrations from scripted AI.

Usage:
    python scripts/collect_demos.py --scenario arena3d --config configs/arena3d/phase1_aim_targets.json --episodes 1000
    python scripts/collect_demos.py --scenario arena3d --config configs/arena3d/phase1_aim_targets.json --episodes 500 --output data/demos/aim.npz --phase 1

    # Drone scenarios (uses cascaded PD expert):
    python scripts/collect_demos.py --scenario drone-hover --config configs/drone/hover.json --episodes 500 --frame-skip 1
    python scripts/collect_demos.py --scenario drone-waypoint --config configs/drone/waypoint.json --episodes 300 --frame-skip 1
    python scripts/collect_demos.py --scenario drone-racing --config configs/drone/racing.json --episodes 200 --frame-skip 1
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser(description="Collect BC demonstrations")
    p.add_argument("--scenario", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--output", default=None)
    p.add_argument("--phase", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=2048,
                   help="Max steps per episode (default 2048)")
    p.add_argument("--frame-skip", type=int, default=4,
                   help="Engine ticks per gym step (default 4)")
    p.add_argument("--filter", default="all", choices=["all", "combat", "aim"],
                   help="Filter demos: all=every tick, combat=shooting ticks, aim=visible enemy ticks")
    args = p.parse_args()

    from training.bc_collector import collect_demonstrations

    output = args.output or f"data/demos/{args.scenario}.npz"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    collect_demonstrations(
        config_path=args.config,
        scenario=args.scenario,
        num_episodes=args.episodes,
        output_path=output,
        phase=args.phase,
        max_steps_per_episode=args.max_steps,
        frame_skip=args.frame_skip,
        filter_mode=args.filter,
    )


if __name__ == "__main__":
    main()
