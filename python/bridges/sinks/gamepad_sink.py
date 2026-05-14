from __future__ import annotations

import gymnasium as gym
import numpy as np

from bridges.core.action_sink import ActionSinkInfo
from bridges.gamepad.protocol import Axis, Button, GamepadBackend
from bridges.gamepad.factory import make_gamepad


class GamepadActionSink:
    """ActionSink adapter wrapping an existing GamepadBackend.

    Maps action array indices to gamepad axes/buttons via configurable mappings.
    The existing GamepadBackend protocol stays untouched.

    Three mapping types:
    - axis_mapping: {action_index: Axis} -- direct 1:1 axis mapping
    - combined_axes: {action_index: (pos_axis, neg_axis)} -- split positive/negative
      (e.g., acceleration -> throttle/brake)
    - button_mapping: {action_index: Button} -- threshold at 0.5 for continuous -> discrete
    """

    def __init__(
        self,
        backend: str = "auto",
        axis_mapping: dict[int, Axis] | None = None,
        combined_axes: dict[int, tuple[Axis, Axis]] | None = None,
        button_mapping: dict[int, Button] | None = None,
        action_space: gym.Space | None = None,
        **backend_kwargs,
    ):
        self._backend_name = backend
        self._backend_kwargs = backend_kwargs
        self._gamepad: GamepadBackend | None = None
        self._axis_mapping = axis_mapping or {}
        self._combined_axes = combined_axes or {}
        self._button_mapping = button_mapping or {}

        n_actions = len(self._axis_mapping) + len(self._combined_axes) + len(self._button_mapping)
        self._action_space = action_space or gym.spaces.Box(
            low=-1.0, high=1.0, shape=(max(n_actions, 1),), dtype=np.float32
        )

    def info(self) -> ActionSinkInfo:
        return ActionSinkInfo(
            name=f"gamepad_{self._backend_name}",
            action_space=self._action_space,
            supports_continuous=True,
            supports_discrete=True,
            platform="windows" if self._backend_name == "vgamepad" else "any",
        )

    def connect(self) -> None:
        self._gamepad = make_gamepad(backend=self._backend_name, **self._backend_kwargs)
        self._gamepad.connect()

    def send(self, action: np.ndarray) -> None:
        if self._gamepad is None:
            raise RuntimeError("GamepadActionSink not connected")

        for idx, axis in self._axis_mapping.items():
            self._gamepad.set_axis(axis, float(action[idx]))

        for idx, (pos_axis, neg_axis) in self._combined_axes.items():
            val = float(action[idx])
            if val >= 0:
                self._gamepad.set_axis(pos_axis, val)
                self._gamepad.set_axis(neg_axis, 0.0)
            else:
                self._gamepad.set_axis(pos_axis, 0.0)
                self._gamepad.set_axis(neg_axis, -val)

        for idx, button in self._button_mapping.items():
            self._gamepad.set_button(button, float(action[idx]) > 0.5)

        self._gamepad.update()

    def reset(self) -> None:
        if self._gamepad is not None:
            self._gamepad.reset()

    def disconnect(self) -> None:
        if self._gamepad is not None:
            self._gamepad.disconnect()
            self._gamepad = None
