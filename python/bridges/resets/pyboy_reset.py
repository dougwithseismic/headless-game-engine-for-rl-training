"""ResetStrategy using PyBoy save states.

Instant, deterministic resets via in-memory save state buffers.
Supports multiple named checkpoints for curriculum learning
(e.g., start from progressively harder game states).
"""

from __future__ import annotations

from bridges.core.reset_strategy import ResetInfo
from bridges.emulators.pyboy_host import PyBoyHost


class PyBoyReset:

    def __init__(
        self,
        host: PyBoyHost,
        initial_state_name: str = "default",
        auto_save_initial: bool = True,
    ):
        self._host = host
        self._active_checkpoint = initial_state_name
        self._auto_save_initial = auto_save_initial
        self._initial_saved = False

    def info(self) -> ResetInfo:
        return ResetInfo(
            name="pyboy_savestate",
            is_instant=True,
            typical_duration_sec=0.001,
            supports_checkpoints=True,
        )

    def reset(self) -> None:
        if not self._initial_saved and self._auto_save_initial:
            self._host.save_state_to_buffer(self._active_checkpoint)
            self._initial_saved = True

        self._host.load_state_from_buffer(self._active_checkpoint)
        self._host.tick(count=4, render=False)
        # Dismiss any active dialogue/textboxes from the save state
        for _ in range(20):
            self._host.button("a", delay=1)
            self._host.tick(count=8, render=False)

    def set_checkpoint(self, checkpoint_id: str) -> None:
        self._active_checkpoint = checkpoint_id

    def save_checkpoint(self, name: str) -> None:
        self._host.save_state_to_buffer(name)
