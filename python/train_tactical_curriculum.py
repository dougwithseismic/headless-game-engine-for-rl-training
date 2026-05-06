"""
4-phase curriculum training for the Tactical Deathmatch scenario.

Phase 1: Aim + shoot in open arena vs scripted AI (~3-5M steps)
Phase 2: Add movement on obstacle maps, load Phase 1 weights (~10-20M steps)
Phase 3: Weapon selection (future, requires multi-weapon) — skipped for now
Phase 4: Self-play league with opponent pool (~20-50M steps)

Usage:
    python train_tactical_curriculum.py
    python train_tactical_curriculum.py --preset dust2
    python train_tactical_curriculum.py --phases 1 2 --live-view
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime


PRESETS = {
    "arena": [
        {
            "name": "p1_aim_shoot",
            "config": "configs/tactical_open.json",
            "timesteps": 3_000_000,
            "scripted_warmup": 3_000_000,
            "description": "Learn to aim and shoot in open arena vs scripted AI",
        },
        {
            "name": "p2_movement",
            "config": "configs/tactical_obstacles.json",
            "timesteps": 10_000_000,
            "scripted_warmup": 2_000_000,
            "description": "Learn movement and cover on obstacle map",
        },
        {
            "name": "p3_selfplay",
            "config": "configs/tactical_obstacles.json",
            "timesteps": 20_000_000,
            "scripted_warmup": 0,
            "description": "Self-play refinement with opponent pool",
        },
        {
            "name": "p4_selfplay_extended",
            "config": "configs/tactical_obstacles.json",
            "timesteps": 20_000_000,
            "scripted_warmup": 0,
            "description": "Extended self-play to push ceiling",
        },
    ],
    "dust2": [
        {
            "name": "p1_aim_shoot",
            "config": "configs/tactical_open.json",
            "timesteps": 3_000_000,
            "scripted_warmup": 3_000_000,
            "description": "Learn to aim and shoot in open arena vs scripted AI",
        },
        {
            "name": "p2_movement",
            "config": "configs/tactical_dust2.json",
            "timesteps": 15_000_000,
            "scripted_warmup": 2_000_000,
            "description": "Learn navigation on dust2 with 25 obstacles",
        },
        {
            "name": "p3_selfplay",
            "config": "configs/tactical_dust2.json",
            "timesteps": 20_000_000,
            "scripted_warmup": 0,
            "description": "Self-play refinement on dust2",
        },
        {
            "name": "p4_selfplay_extended",
            "config": "configs/tactical_dust2.json",
            "timesteps": 20_000_000,
            "scripted_warmup": 0,
            "description": "Extended self-play to push ceiling",
        },
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="Tactical Curriculum Training")
    p.add_argument("--preset", default="arena", choices=list(PRESETS.keys()))
    p.add_argument("--phases", nargs="*", type=int, default=None,
                   help="Run only specific phases (1-indexed), e.g. --phases 1 2")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--live-view", action="store_true")
    p.add_argument("--live-port", type=int, default=3000)
    return p.parse_args()


def resolve_config(path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, path) if not os.path.isabs(path) else path


def find_best_model(run_dir):
    best = os.path.join(run_dir, "best_model", "best_model.zip")
    if os.path.exists(best):
        return best
    final = os.path.join(run_dir, "final_model.zip")
    if os.path.exists(final):
        return final
    return None


def run_phase(phase_idx, phase, run_base, resume_from, args):
    phase_num = phase_idx + 1
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    phase_name = f"{phase['name']}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"  PHASE {phase_num}: {phase['description']}")
    print(f"  Config: {phase['config']}")
    print(f"  Steps: {phase['timesteps']:,}")
    print(f"  Warmup: {phase['scripted_warmup']:,}")
    if resume_from:
        print(f"  Resume: {resume_from}")
    print(f"{'='*60}\n")

    config_path = resolve_config(phase["config"])

    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "train_tactical.py"),
        "--config", config_path,
        "--timesteps", str(phase["timesteps"]),
        "--scripted-warmup", str(phase["scripted_warmup"]),
        "--n-envs", str(args.n_envs),
        "--name", phase_name,
        "--eval-freq", "100000",
        "--checkpoint-freq", "1000000",
    ]

    if resume_from:
        cmd.extend(["--resume", resume_from])

    if args.live_view:
        cmd.extend(["--live-view", "--live-port", str(args.live_port)])

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n  PHASE {phase_num} FAILED (exit code {result.returncode})")
        return None

    print(f"\n  Phase {phase_num} complete in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Find the best model from this phase
    run_dir = None
    runs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")
    if os.path.isdir(runs_path):
        candidates = sorted(
            [d for d in os.listdir(runs_path) if d.startswith(phase["name"])],
            reverse=True,
        )
        if candidates:
            run_dir = os.path.join(runs_path, candidates[0])

    if run_dir:
        model_path = find_best_model(run_dir)
        print(f"  Best model: {model_path}")
        return model_path

    return None


def main():
    args = parse_args()
    phases = PRESETS[args.preset]

    if args.phases:
        phase_indices = [p - 1 for p in args.phases if 0 < p <= len(phases)]
    else:
        phase_indices = list(range(len(phases)))

    print(f"=== Tactical Curriculum: {args.preset} preset ===")
    print(f"Phases to run: {[i+1 for i in phase_indices]}")
    print(f"Envs: {args.n_envs}")

    total_steps = sum(phases[i]["timesteps"] for i in phase_indices)
    print(f"Total steps: {total_steps:,}")

    run_base = os.path.join("runs", f"curriculum_{args.preset}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(run_base, exist_ok=True)

    with open(os.path.join(run_base, "curriculum.json"), "w") as f:
        json.dump({
            "preset": args.preset,
            "phases": phases,
            "phase_indices": phase_indices,
            "n_envs": args.n_envs,
        }, f, indent=2)

    resume_from = None
    results = []

    for idx in phase_indices:
        phase = phases[idx]
        model_path = run_phase(idx, phase, run_base, resume_from, args)

        results.append({
            "phase": idx + 1,
            "name": phase["name"],
            "model": model_path,
            "success": model_path is not None,
        })

        if model_path is None:
            print(f"\nStopping curriculum: Phase {idx+1} failed.")
            break

        resume_from = model_path

    print(f"\n{'='*60}")
    print(f"  CURRICULUM COMPLETE")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"  Phase {r['phase']} ({r['name']}): {status}")
        if r["model"]:
            print(f"    Model: {r['model']}")
    print()


if __name__ == "__main__":
    main()
