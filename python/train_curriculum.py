"""
Auto-curriculum training pipeline.

Runs sequential phases, each resuming from the previous best model.
Keeps going as long as phases improve. Logs everything.

Usage:
    python python/train_curriculum.py
    python python/train_curriculum.py --live-view
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHASE_MAPS = {
    "arena": [
        {
            "name": "p1_combat",
            "config": "configs/1v1_open.json",
            "timesteps": 3_000_000,
            "scripted_warmup": 1_000_000,
            "desc": "Open arena — learn to aim, shoot, dodge",
        },
        {
            "name": "p2_obstacles",
            "config": "configs/1v1_deathmatch.json",
            "timesteps": 5_000_000,
            "scripted_warmup": 500_000,
            "desc": "Add obstacles — learn navigation + cover",
        },
        {
            "name": "p3_selfplay_long",
            "config": "configs/1v1_deathmatch.json",
            "timesteps": 10_000_000,
            "scripted_warmup": 0,
            "desc": "Extended self-play — refine against strong opponents",
        },
        {
            "name": "p4_selfplay_extended",
            "config": "configs/1v1_deathmatch.json",
            "timesteps": 10_000_000,
            "scripted_warmup": 0,
            "desc": "More self-play — push the ceiling",
        },
    ],
    "dust2": [
        {
            "name": "p1_combat",
            "config": "configs/1v1_dust2_open.json",
            "timesteps": 3_000_000,
            "scripted_warmup": 1_000_000,
            "desc": "Open 1200x1200 — learn to aim, shoot, dodge at range",
        },
        {
            "name": "p2_obstacles",
            "config": "configs/1v1_dust2.json",
            "timesteps": 5_000_000,
            "scripted_warmup": 500_000,
            "desc": "Dust 2 corridors — learn navigation + cover + LOS",
        },
        {
            "name": "p3_selfplay_long",
            "config": "configs/1v1_dust2.json",
            "timesteps": 10_000_000,
            "scripted_warmup": 0,
            "desc": "Extended self-play on Dust 2",
        },
        {
            "name": "p4_selfplay_extended",
            "config": "configs/1v1_dust2.json",
            "timesteps": 10_000_000,
            "scripted_warmup": 0,
            "desc": "More self-play — push the ceiling on Dust 2",
        },
    ],
}


def find_latest_model(run_pattern):
    candidates = []
    for path in glob.glob(os.path.join(PROJECT_ROOT, "runs", run_pattern, "best_model", "best_model.zip")):
        candidates.append(path)
    for path in glob.glob(os.path.join(PROJECT_ROOT, "runs", run_pattern, "final_model.zip")):
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime).replace(".zip", "")


def run_phase(phase, resume_from=None, live_view=False, live_port=3000, recurrent=False):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"auto_{phase['name']}_{timestamp}"

    config_path = os.path.join(PROJECT_ROOT, phase["config"])

    if recurrent:
        n_steps, batch_size, n_epochs = "128", "128", "5"
    else:
        n_steps, batch_size, n_epochs = "2048", "256", "10"

    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "python", "train_selfplay.py"),
        "--config", config_path,
        "--scenario", "fps",
        "--timesteps", str(phase["timesteps"]),
        "--n-envs", "16",
        "--frame-skip", "4",
        "--max-steps", "2048",
        "--lr", "3e-4",
        "--batch-size", batch_size,
        "--n-steps", n_steps,
        "--n-epochs", n_epochs,
        "--gamma", "0.99",
        "--ent-coef", "0.01",
        "--swap-interval", "250000",
        "--scripted-warmup", str(phase["scripted_warmup"]),
        "--eval-freq", "250000",
        "--checkpoint-freq", "1000000",
        "--name", run_name,
        "--no-normalize",
    ]

    if recurrent:
        cmd.append("--recurrent")

    if resume_from:
        cmd.extend(["--resume", resume_from])

    if live_view:
        cmd.extend(["--live-view", "--live-port", str(live_port)])

    print(f"\n{'='*70}")
    print(f"  Phase: {phase['name']}")
    print(f"  {phase['desc']}")
    print(f"  Config: {phase['config']}")
    print(f"  Steps: {phase['timesteps']:,}")
    print(f"  Resume: {resume_from or 'fresh'}")
    print(f"  Run: {run_name}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0

    print(f"\n  Phase {phase['name']} finished in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Exit code: {result.returncode}")

    best = find_latest_model(f"{run_name}*")
    if best:
        print(f"  Best model: {best}")
    else:
        print(f"  WARNING: No model found for {run_name}")

    return best, result.returncode


def main():
    parser = argparse.ArgumentParser(description="Auto-curriculum training pipeline")
    parser.add_argument("--live-view", action="store_true")
    parser.add_argument("--live-port", type=int, default=3000)
    parser.add_argument("--start-phase", type=int, default=0, help="Phase index to start from (0-based)")
    parser.add_argument("--resume", default=None, help="Model to resume from for the first phase")
    parser.add_argument("--recurrent", action="store_true", help="Use RecurrentPPO (LSTM)")
    parser.add_argument("--map", default="arena", choices=list(PHASE_MAPS.keys()),
                       help="Map preset to use (arena or dust2)")
    args = parser.parse_args()

    PHASES = PHASE_MAPS[args.map]

    print(f"GhostLobby Auto-Curriculum Training")
    print(f"Map: {args.map}")
    print(f"Phases: {len(PHASES)}")
    print(f"Started: {datetime.now().isoformat()}")

    resume_from = args.resume

    for i, phase in enumerate(PHASES):
        if i < args.start_phase:
            pattern = f"auto_{phase['name']}_*"
            prev = find_latest_model(pattern)
            if prev:
                resume_from = prev
                print(f"  Skipping phase {i} ({phase['name']}), using existing: {prev}")
            continue

        best, exit_code = run_phase(
            phase,
            resume_from=resume_from,
            live_view=args.live_view,
            live_port=args.live_port,
            recurrent=args.recurrent,
        )

        if exit_code != 0:
            print(f"\n  Phase {phase['name']} FAILED (exit {exit_code}). Stopping.")
            break

        if best:
            resume_from = best
        else:
            print(f"\n  No model produced. Stopping.")
            break

    print(f"\nPipeline complete. Final model: {resume_from}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
