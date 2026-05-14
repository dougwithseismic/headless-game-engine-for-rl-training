from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class TimingPolicy(str, Enum):
    REAL_TIME = "real_time"        # Sleep to maintain target_hz
    FREE_RUNNING = "free_running"  # Step as fast as possible
    FIXED_STEP = "fixed_step"      # External clock / hardware timing


@dataclass
class TimingConfig:
    policy: TimingPolicy
    target_hz: float = 20.0
    frame_skip: int = 1


class StepTimer:
    """Enforces timing policy between steps."""

    def __init__(self, config: TimingConfig):
        self.config = config
        self._last_step_time: float | None = None

    def wait(self) -> None:
        if self.config.policy == TimingPolicy.FREE_RUNNING:
            return
        if self.config.policy == TimingPolicy.FIXED_STEP:
            raise NotImplementedError("FIXED_STEP requires an external clock — not yet implemented")
        now = time.monotonic()
        if self._last_step_time is not None:
            dt_target = 1.0 / self.config.target_hz
            elapsed = now - self._last_step_time
            remaining = dt_target - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_step_time = time.monotonic()

    def reset(self) -> None:
        self._last_step_time = None
