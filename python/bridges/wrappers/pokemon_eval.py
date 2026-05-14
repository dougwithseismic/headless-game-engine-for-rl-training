"""Honest Pokemon eval callback: stochastic policy, real game progress metrics.

Runs eval episodes with deterministic=False and logs actual game progress,
not just reward. Tracks milestone completion, real movement, battle outcomes.

Usage:
    callback = PokemonEvalCallback(eval_env, eval_freq=10000)
    model.learn(total_timesteps=500000, callback=[callback])
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class PokemonEvalCallback(BaseCallback):
    """Evaluate with stochastic policy and log honest game progress."""

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: str | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = Path(log_path) if log_path else None
        self._eval_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        self._eval_count += 1
        results = []

        for ep in range(self.n_eval_episodes):
            result = self._run_episode()
            results.append(result)

        # Aggregate
        numeric_keys = [k for k in results[0] if k != "reached_route_29"]
        avg = {}
        for k in numeric_keys:
            vals = [float(r.get(k, 0)) for r in results]
            avg[k] = np.mean(vals)
        any_route29 = any(r.get("reached_route_29", False) for r in results)

        # Log to TensorBoard
        for key, val in avg.items():
            self.logger.record(f"eval_honest/{key}", float(val))
        self.logger.record("eval_honest/any_route_29", int(any_route29))

        if self.verbose:
            print(f"\n  [Honest Eval @ {self.num_timesteps}] "
                  f"tiles={avg.get('unique_tiles',0):.0f} maps={avg.get('maps_reached',0):.1f} "
                  f"moved={avg.get('move_pct',0):.0f}% route29={'YES' if any_route29 else 'no'} "
                  f"battles={avg.get('battles',0):.1f} reward={avg.get('total_reward',0):.1f}")

        # Save detailed log
        if self.log_path:
            self.log_path.mkdir(parents=True, exist_ok=True)
            log_file = self.log_path / f"eval_{self.num_timesteps:08d}.json"
            with open(log_file, "w") as f:
                json.dump({"timestep": self.num_timesteps, "episodes": results, "avg": avg}, f, indent=2, default=str)

        return True

    def _run_episode(self) -> dict:
        obs = self.eval_env.reset()
        positions = []
        maps_seen = set()
        battles = 0
        battles_won = 0
        total_reward = 0.0
        steps = 0
        steps_moved = 0
        reached_route_29 = False
        milestones_hit = []

        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=False)
            obs, reward, done_arr, info = self.eval_env.step(action)
            done = done_arr[0]
            total_reward += reward[0]
            steps += 1

            # Extract position from info or obs
            if "game_stats" in (info[0] if info else {}):
                stats = info[0]["game_stats"]

            # Read from obs — need to handle both Dict and flat obs
            ram = self._get_ram(obs)
            if ram is not None:
                mg = int(round(ram[0] * 255))  # map_group at index 0...
                # Actually we need the feature index. Let's use game_stats from info instead.
                pass

            if done and info and "game_stats" in info[0]:
                stats = info[0]["game_stats"]
                return {
                    "unique_tiles": stats.get("tiles_visited", 0),
                    "maps_reached": stats.get("maps_visited", 0),
                    "battles": stats.get("wild_battles", 0) + stats.get("trainer_battles", 0),
                    "battles_won": stats.get("battles_won", 0),
                    "level_ups": stats.get("level_ups", 0),
                    "badges": stats.get("badges_earned", 0),
                    "steps_moved_pct": 100 * (1 - stats.get("steps_stuck", 0) / max(stats.get("total_steps", 1), 1)),
                    "steps_in_battle": stats.get("steps_in_battle", 0),
                    "steps_in_menu": stats.get("steps_in_menu", 0),
                    "total_reward": total_reward,
                    "total_steps": steps,
                    "move_pct": 100 * (1 - stats.get("steps_stuck", 0) / max(stats.get("total_steps", 1), 1)),
                    "reached_route_29": stats.get("maps_visited", 0) >= 2,
                }

        return {
            "unique_tiles": 0, "maps_reached": 0, "battles": 0, "battles_won": 0,
            "level_ups": 0, "badges": 0, "steps_moved_pct": 0, "steps_in_battle": 0,
            "steps_in_menu": 0, "total_reward": total_reward, "total_steps": steps,
            "move_pct": 0, "reached_route_29": False,
        }

    def _get_ram(self, obs):
        if isinstance(obs, dict) and "ram" in obs:
            return obs["ram"][0]
        return None
