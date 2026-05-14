from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class Axis(str, Enum):
    """Named axes on an Xbox 360 controller."""

    LEFT_STICK_X = "left_stick_x"
    LEFT_STICK_Y = "left_stick_y"
    RIGHT_STICK_X = "right_stick_x"
    RIGHT_STICK_Y = "right_stick_y"
    LEFT_TRIGGER = "left_trigger"
    RIGHT_TRIGGER = "right_trigger"


class Button(str, Enum):
    """Named buttons on an Xbox 360 controller."""

    A = "a"
    B = "b"
    X = "x"
    Y = "y"
    LB = "lb"
    RB = "rb"
    START = "start"
    BACK = "back"
    L3 = "l3"
    R3 = "r3"
    DPAD_UP = "dpad_up"
    DPAD_DOWN = "dpad_down"
    DPAD_LEFT = "dpad_left"
    DPAD_RIGHT = "dpad_right"


# Axes that use [-1, 1] range (sticks) vs [0, 1] range (triggers).
STICK_AXES = frozenset(
    {Axis.LEFT_STICK_X, Axis.LEFT_STICK_Y, Axis.RIGHT_STICK_X, Axis.RIGHT_STICK_Y}
)
TRIGGER_AXES = frozenset({Axis.LEFT_TRIGGER, Axis.RIGHT_TRIGGER})

AXIS_NAMES: list[str] = [a.value for a in Axis]
BUTTON_NAMES: list[str] = [b.value for b in Button]

DEFAULT_DEADZONE = 0.05


def apply_deadzone(value: float, deadzone: float, axis: Axis) -> float:
    """Apply deadzone then clamp to the valid range for *axis*.

    For sticks ([-1, 1]): values within [-deadzone, +deadzone] snap to 0,
    then the remaining range is rescaled so the output still reaches -1/+1
    at the physical extremes.

    For triggers ([0, 1]): values below deadzone snap to 0, remaining range
    rescaled to [0, 1].
    """
    if axis in STICK_AXES:
        if abs(value) < deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        scaled = (abs(value) - deadzone) / (1.0 - deadzone)
        return max(-1.0, min(1.0, sign * scaled))
    # Trigger
    if value < deadzone:
        return 0.0
    scaled = (value - deadzone) / (1.0 - deadzone)
    return max(0.0, min(1.0, scaled))


def coerce_axis(axis: Axis | str) -> Axis:
    if isinstance(axis, Axis):
        return axis
    try:
        return Axis(axis)
    except ValueError:
        raise ValueError(f"Unknown axis: {axis!r}. Valid: {[a.value for a in Axis]}")


def coerce_button(button: Button | str) -> Button:
    if isinstance(button, Button):
        return button
    try:
        return Button(button)
    except ValueError:
        raise ValueError(f"Unknown button: {button!r}. Valid: {[b.value for b in Button]}")


@dataclass
class GamepadState:
    """Full snapshot of controller state — all axes and buttons."""

    axes: dict[Axis, float] = field(default_factory=lambda: {a: 0.0 for a in Axis})
    buttons: dict[Button, bool] = field(
        default_factory=lambda: {b: False for b in Button}
    )

    def reset(self) -> None:
        """Zero all axes, release all buttons."""
        for a in Axis:
            self.axes[a] = 0.0
        for b in Button:
            self.buttons[b] = False

    def clone(self) -> GamepadState:
        return GamepadState(
            axes=dict(self.axes),
            buttons=dict(self.buttons),
        )


@runtime_checkable
class GamepadBackend(Protocol):
    """Interface for a virtual Xbox 360-style gamepad.

    Implementations: vgamepad (Windows/ViGEmBus), mock (Mac/testing), etc.
    """

    def connect(self) -> None:
        """Initialize the virtual controller."""
        ...

    def disconnect(self) -> None:
        """Release the virtual controller."""
        ...

    def set_axis(self, axis: Axis, value: float) -> None:
        """Set a controller axis.

        Sticks (LEFT_STICK_*, RIGHT_STICK_*): value in [-1, 1].
        Triggers (LEFT_TRIGGER, RIGHT_TRIGGER): value in [0, 1].
        """
        ...

    def set_button(self, button: Button, pressed: bool) -> None:
        """Press or release a button."""
        ...

    def update(self) -> None:
        """Flush all pending state to the virtual controller."""
        ...

    def reset(self) -> None:
        """Zero all axes, release all buttons, and flush."""
        ...
