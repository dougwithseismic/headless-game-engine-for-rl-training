#!/usr/bin/env python3
"""Eureka-style LLM-guided reward search.

Uses Claude to generate reward configurations, runs each as a short
training candidate, feeds results back to Claude for the next round.

Usage:
    python scripts/eureka_search.py \
        --base-config configs/cs_lite/1v1_wide.json \
        --scenario cs_lite_dummy \
        --bc-model data/bc_models/cs_lite_dummy.zip \
        --candidates 4 \
        --rounds 3 \
        --steps 500000
"""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_CLI_API_KEY = None

REWARD_KEYS = [
    "reward_kill", "reward_death", "reward_damage_dealt", "reward_damage_taken",
    "reward_round_win", "reward_round_loss", "reward_near_miss",
    "reward_friendly_fire", "reward_bomb_plant", "reward_bomb_defuse",
    "reward_bomb_pickup",
]

SYSTEM_PROMPT = """You are designing reward functions for an FPS game AI agent.

The agent plays a 1v1 Counter-Strike-style game with bomb plant/defuse mechanics.
It uses reinforcement learning (PPO) and learns from the reward signal you design.

Available reward keys (all floats, can be positive or negative):
- reward_kill: reward for killing an enemy
- reward_death: reward for dying (typically 0 or negative)
- reward_damage_dealt: reward per hit on enemy (damage is 25 per hit)
- reward_damage_taken: reward per hit taken (typically 0 or negative)
- reward_round_win: reward for winning a round
- reward_round_loss: reward for losing a round (typically 0 or negative)
- reward_near_miss: reward for shots that nearly hit (encourages aim practice)
- reward_friendly_fire: penalty for hitting teammates (negative)
- reward_bomb_plant: reward for planting the bomb at a site
- reward_bomb_defuse: reward for defusing the planted bomb
- reward_bomb_pickup: reward for picking up a dropped bomb

Key behavioral metrics we measure:
- accuracy: % of shots that hit (higher = better aim, 20-30% is typical, 80%+ is excellent)
- kills_per_ep: kills per episode (higher = more lethal)
- deaths_per_ep: deaths per episode (lower = better survival)
- shoot_rate: % of timesteps agent fires (10-20% is healthy, >30% is spray-and-pray)
- win_rate: % of rounds won
- damage_dealt_per_ep: total damage output per episode

Important learnings from previous research:
- Positive-only rewards (no death penalty) work best for Phase 1 training
- Heavy death penalties cause the agent to avoid combat entirely
- Near-miss rewards help early aim learning
- Bomb plant/defuse rewards encourage objective play beyond just fighting
- The agent has auto-aim — accuracy depends on positioning and fire timing, not aim control

Respond with ONLY a JSON array of reward configs. Each config is an object with the reward keys above.
No markdown, no explanation, just the JSON array."""


def ask_claude(history: list, n_candidates: int) -> list[dict]:
    """Call Claude to generate reward configs based on previous results."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY") or _CLI_API_KEY
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set and --api-key not provided")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    user_msg = ""
    if not history:
        user_msg = f"""This is the first round. Generate {n_candidates} diverse reward configurations.
Include at least:
1. A positive-only config (no penalties)
2. A balanced config (some penalties)
3. An objective-focused config (high bomb plant/defuse rewards)
4. An aggressive combat config (high kill rewards)"""
    else:
        user_msg = f"Here are the results from previous rounds:\n\n"
        for round_idx, round_data in enumerate(history):
            user_msg += f"=== Round {round_idx + 1} ===\n"
            for r in sorted(round_data, key=lambda x: -x.get("accuracy", 0)):
                user_msg += (
                    f"  {r['name']}: accuracy={r['accuracy']:.1%}, "
                    f"kills={r['kills_per_ep']:.1f}, deaths={r['deaths_per_ep']:.1f}, "
                    f"win_rate={r.get('win_rate', 0):.1%}, shoot_rate={r['shoot_rate']:.1%}, "
                    f"reward={r['reward']:+.1f}\n"
                    f"    config: {json.dumps(r['overrides'])}\n"
                )
            user_msg += "\n"
        user_msg += f"""Based on these results, generate {n_candidates} new reward configs.
Mutate and combine the best performers. Try to improve accuracy and kills.
Avoid repeating configs that produced low accuracy or 0 kills.
Explore at least one novel strategy not yet tried."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    configs = json.loads(text)
    if not isinstance(configs, list):
        configs = [configs]
    return configs


def generate_fallback_configs(n: int) -> list[dict]:
    """Fallback configs if Claude API isn't available."""
    configs = [
        {"reward_kill": 5.0, "reward_death": 0.0, "reward_damage_dealt": 2.0,
         "reward_damage_taken": 0.0, "reward_round_win": 5.0, "reward_round_loss": 0.0,
         "reward_near_miss": 0.05, "reward_bomb_plant": 3.0, "reward_bomb_defuse": 3.0},

        {"reward_kill": 3.0, "reward_death": -1.0, "reward_damage_dealt": 1.0,
         "reward_damage_taken": -0.5, "reward_round_win": 5.0, "reward_round_loss": -1.0,
         "reward_near_miss": 0.03, "reward_bomb_plant": 5.0, "reward_bomb_defuse": 5.0},

        {"reward_kill": 8.0, "reward_death": 0.0, "reward_damage_dealt": 3.0,
         "reward_damage_taken": 0.0, "reward_round_win": 10.0, "reward_round_loss": 0.0,
         "reward_near_miss": 0.1, "reward_bomb_plant": 1.0, "reward_bomb_defuse": 1.0},

        {"reward_kill": 3.0, "reward_death": 0.0, "reward_damage_dealt": 1.0,
         "reward_damage_taken": 0.0, "reward_round_win": 3.0, "reward_round_loss": 0.0,
         "reward_near_miss": 0.03, "reward_bomb_plant": 10.0, "reward_bomb_defuse": 10.0},
    ]
    return configs[:n]


def run_candidate(scenario, config_path, bc_model, steps, n_envs, run_name):
    """Run a single training candidate and return behavioral metrics."""
    from training.ppo_trainer import PPOTrainer

    trainer = PPOTrainer(
        scenario=scenario,
        config_path=config_path,
        name=run_name,
        timesteps=steps,
        n_envs=n_envs,
        n_eval_episodes=10,
        eval_freq=steps,
        checkpoint_freq=steps + 1,
        resume=bc_model,
        auto_stop=False,
        track_behavior=True,
        n_steps=2048,
        batch_size=256,
    )

    trainer.train()

    metrics = {"reward": 0.0, "accuracy": 0.0, "kills_per_ep": 0.0,
               "deaths_per_ep": 0.0, "shoot_rate": 0.0, "damage_dealt_per_ep": 0.0,
               "win_rate": 0.0}

    try:
        import glob
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tb_files = glob.glob(os.path.join(trainer.run_dir, "tb", "**", "events*"), recursive=True)
        if tb_files:
            ea = EventAccumulator(tb_files[0])
            ea.Reload()
            tag_map = {
                "eval/mean_reward": "reward",
                "behavior/accuracy": "accuracy",
                "behavior/kills_per_ep": "kills_per_ep",
                "behavior/deaths_per_ep": "deaths_per_ep",
                "behavior/shoot_rate": "shoot_rate",
                "behavior/damage_dealt_per_ep": "damage_dealt_per_ep",
                "behavior/win_rate": "win_rate",
            }
            for tb_tag, key in tag_map.items():
                if tb_tag in ea.Tags().get("scalars", []):
                    vals = ea.Scalars(tb_tag)
                    if vals:
                        metrics[key] = float(vals[-1].value)
    except Exception:
        pass

    return metrics


def main():
    p = argparse.ArgumentParser(description="Eureka-style LLM-guided reward search")
    p.add_argument("--base-config", required=True)
    p.add_argument("--scenario", default="cs_lite_dummy")
    p.add_argument("--bc-model", default=None)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--no-llm", action="store_true", help="Use fallback configs instead of Claude")
    p.add_argument("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    p.add_argument("--output", default="data/eureka_results.json")
    args = p.parse_args()

    global _CLI_API_KEY
    _CLI_API_KEY = args.api_key

    print(f"=== Eureka Reward Search ===")
    print(f"  Candidates per round: {args.candidates}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Steps per candidate: {args.steps:,}")
    print(f"  Total training: {args.candidates * args.rounds * args.steps:,} steps")
    print(f"  BC warm-start: {args.bc_model}")
    print()

    all_history = []

    for round_idx in range(args.rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {round_idx + 1}/{args.rounds}")
        print(f"{'='*60}")

        if args.no_llm:
            configs = generate_fallback_configs(args.candidates)
        else:
            print("Asking Claude for reward configs...", flush=True)
            configs = ask_claude(all_history, args.candidates)

        round_results = []
        for i, config in enumerate(configs):
            name = f"r{round_idx+1}_c{i+1}"
            print(f"\n[{i+1}/{len(configs)}] {name}: {json.dumps(config)}", flush=True)

            tmp_config = os.path.join(tempfile.gettempdir(), f"eureka_{name}.json")
            with open(args.base_config) as f:
                base = json.load(f)
            for k, v in config.items():
                base.setdefault("extra", {})[k] = v
            with open(tmp_config, "w") as f:
                json.dump(base, f, indent=2)

            t0 = time.time()
            metrics = run_candidate(
                scenario=args.scenario,
                config_path=tmp_config,
                bc_model=args.bc_model,
                steps=args.steps,
                n_envs=args.n_envs,
                run_name=f"eureka_{name}",
            )
            elapsed = time.time() - t0

            metrics["name"] = name
            metrics["overrides"] = config
            metrics["elapsed_s"] = elapsed
            round_results.append(metrics)

            print(f"  {elapsed:.0f}s: acc={metrics['accuracy']:.1%} "
                  f"kills={metrics['kills_per_ep']:.1f} win={metrics['win_rate']:.1%} "
                  f"reward={metrics['reward']:+.1f}", flush=True)

        all_history.append(round_results)

        print(f"\n--- Round {round_idx+1} Summary ---")
        for r in sorted(round_results, key=lambda x: -x["accuracy"]):
            print(f"  {r['name']}: acc={r['accuracy']:.1%} kills={r['kills_per_ep']:.1f} "
                  f"win={r['win_rate']:.1%} reward={r['reward']:+.1f}")

    # Final summary
    print(f"\n{'='*60}")
    print("EUREKA SEARCH COMPLETE")
    print(f"{'='*60}")

    all_results = [r for rnd in all_history for r in rnd]
    best = max(all_results, key=lambda r: r["accuracy"] + r["kills_per_ep"] * 0.1)
    print(f"\nBest config: {best['name']}")
    print(f"  accuracy={best['accuracy']:.1%} kills={best['kills_per_ep']:.1f} "
          f"win={best['win_rate']:.1%} reward={best['reward']:+.1f}")
    print(f"  config: {json.dumps(best['overrides'], indent=2)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"rounds": all_history, "best": best}, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
