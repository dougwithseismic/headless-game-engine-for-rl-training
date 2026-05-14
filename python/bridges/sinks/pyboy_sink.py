"""ActionSink for Game Boy games via PyBoy.

Maps a MultiBinary(8) action array to Game Boy button presses:
  [up, down, left, right, a, b, start, select]

Each element: 1 = pressed, 0 = released.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from bridges.core.action_sink import ActionSinkInfo
from bridges.emulators.pyboy_host import PyBoyHost, GAMEBOY_BUTTONS


class PyBoyActionSink:

    def __init__(self, host: PyBoyHost, buttons: tuple[str, ...] | None = None):
        self._host = host
        self._buttons = buttons or GAMEBOY_BUTTONS
        self._held: set[str] = set()

    def info(self) -> ActionSinkInfo:
        return ActionSinkInfo(
            name="pyboy",
            action_space=gym.spaces.MultiBinary(len(self._buttons)),
            supports_continuous=False,
            supports_discrete=True,
            platform="any",
        )

    def connect(self) -> None:
        if not self._host.started:
            self._host.start()

    def send(self, action: np.ndarray) -> None:
        for i, button_name in enumerate(self._buttons):
            pressed = bool(action[i])
            was_held = button_name in self._held

            if pressed and not was_held:
                self._host.button_press(button_name)
                self._held.add(button_name)
            elif not pressed and was_held:
                self._host.button_release(button_name)
                self._held.discard(button_name)

    def reset(self) -> None:
        for button_name in list(self._held):
            self._host.button_release(button_name)
        self._held.clear()

    def disconnect(self) -> None:
        self.reset()
