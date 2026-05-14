"""SB3 callbacks for Pokemon Gold training telemetry.

Logs game-specific metrics to TensorBoard: tiles explored, battles,
level ups, badges, exploration rate, etc.

Usage:
    callbacks = [GoldTelemetryCallback()]
    model.learn(total_timesteps=100000, callback=callbacks)
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class GoldTelemetryCallback(BaseCallback):
    """Log Pokemon game stats to TensorBoard at episode boundaries."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            stats = info.get("game_stats")
            if stats is None:
                continue

            self._episode_count += 1
            prefix = "pokemon"

            self.logger.record(f"{prefix}/tiles_visited", stats["tiles_visited"])
            self.logger.record(f"{prefix}/maps_visited", stats["maps_visited"])
            self.logger.record(f"{prefix}/exploration_rate", stats["exploration_rate"])
            self.logger.record(f"{prefix}/battles_won", stats["battles_won"])
            self.logger.record(f"{prefix}/battles_lost", stats["battles_lost"])
            self.logger.record(f"{prefix}/wild_battles", stats["wild_battles"])
            self.logger.record(f"{prefix}/trainer_battles", stats["trainer_battles"])
            self.logger.record(f"{prefix}/level_ups", stats["level_ups"])
            self.logger.record(f"{prefix}/max_level", stats["max_level"])
            self.logger.record(f"{prefix}/badges_earned", stats["badges_earned"])
            self.logger.record(f"{prefix}/party_size_max", stats["party_size_max"])
            self.logger.record(f"{prefix}/steps_in_battle", stats["steps_in_battle"])
            self.logger.record(f"{prefix}/steps_in_menu", stats["steps_in_menu"])
            self.logger.record(f"{prefix}/steps_stuck", stats["steps_stuck"])

            events = info.get("game_events", [])
            if events and self.verbose > 0:
                for e in events:
                    print(f"  [{e['type']}] step={e['step']} {e['data']}")

        return True
