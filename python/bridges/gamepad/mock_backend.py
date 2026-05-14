from __future__ import annotations

import time
from collections import deque

from bridges.gamepad.protocol import (
    Axis,
    Button,
    DEFAULT_DEADZONE,
    GamepadBackend,
    GamepadState,
    STICK_AXES,
    TRIGGER_AXES,
    apply_deadzone,
)


class TimestampedState:
    """A gamepad state snapshot paired with a timestamp."""

    __slots__ = ("state", "timestamp")

    def __init__(self, state: GamepadState, timestamp: float) -> None:
        self.state = state
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return f"TimestampedState(t={self.timestamp:.4f}, state={self.state})"


def _coerce_axis(axis: Axis | str) -> Axis:
    if isinstance(axis, Axis):
        return axis
    try:
        return Axis(axis)
    except ValueError:
        raise ValueError(f"Unknown axis: {axis!r}. Valid: {[a.value for a in Axis]}")


def _coerce_button(button: Button | str) -> Button:
    if isinstance(button, Button):
        return button
    try:
        return Button(button)
    except ValueError:
        raise ValueError(f"Unknown button: {button!r}. Valid: {[b.value for b in Button]}")


def _clamp_axis(axis: Axis, value: float) -> float:
    if axis in STICK_AXES:
        return max(-1.0, min(1.0, value))
    return max(0.0, min(1.0, value))


class MockGamepadBackend:
    """In-process mock Xbox 360 gamepad for Mac development and testing.

    Satisfies the :class:`GamepadBackend` protocol without any OS-level
    virtual-device driver.  Every :meth:`update` call snapshots the current
    state into a ring buffer so tests can assert on the full history of
    inputs that were flushed.
    """

    def __init__(self, *, max_history: int = 1000, deadzone: float = DEFAULT_DEADZONE) -> None:
        self._state = GamepadState()
        self._connected = False
        self._history: deque[TimestampedState] = deque(maxlen=max_history)
        self._update_count = 0
        self.deadzone = deadzone

    # -- protocol methods --------------------------------------------------

    def connect(self) -> None:
        if self._connected:
            raise RuntimeError("Already connected")
        self._connected = True

    def disconnect(self) -> None:
        self._require_connected()
        self._connected = False

    def set_axis(self, axis: Axis | str, value: float) -> None:
        self._require_connected()
        axis = _coerce_axis(axis)
        self._state.axes[axis] = apply_deadzone(value, self.deadzone, axis)

    def set_button(self, button: Button | str, pressed: bool) -> None:
        self._require_connected()
        button = _coerce_button(button)
        self._state.buttons[button] = pressed

    def update(self) -> None:
        self._require_connected()
        self._history.append(TimestampedState(self._state.clone(), time.monotonic()))
        self._update_count += 1

    def reset(self) -> None:
        self._require_connected()
        self._state.reset()
        self.update()

    # -- convenience --------------------------------------------------------

    def get_state(self) -> GamepadState:
        """Return the current (live, pre-flush) gamepad state."""
        return self._state.clone()

    def get_history(self) -> list[TimestampedState]:
        """Return all recorded snapshots oldest-first."""
        return list(self._history)

    @property
    def update_count(self) -> int:
        """Total number of :meth:`update` calls since creation."""
        return self._update_count

    @property
    def connected(self) -> bool:
        return self._connected

    def print_state(self) -> None:
        """Print a compact ASCII picture of the gamepad to the terminal."""
        s = self._state
        lx = s.axes[Axis.LEFT_STICK_X]
        ly = s.axes[Axis.LEFT_STICK_Y]
        rx = s.axes[Axis.RIGHT_STICK_X]
        ry = s.axes[Axis.RIGHT_STICK_Y]
        lt = s.axes[Axis.LEFT_TRIGGER]
        rt = s.axes[Axis.RIGHT_TRIGGER]

        def _btn(b: Button) -> str:
            return b.value.upper() if s.buttons[b] else "."

        def _stick(x: float, y: float) -> str:
            return f"({x:+.2f},{y:+.2f})"

        def _bar(v: float, width: int = 5) -> str:
            filled = round(v * width)
            return "#" * filled + "-" * (width - filled)

        lines = [
            f"  LT [{_bar(lt)}] {lt:.2f}          RT [{_bar(rt)}] {rt:.2f}",
            f"  {_btn(Button.LB):>2}                          {_btn(Button.RB)}",
            "",
            f"       {_btn(Button.DPAD_UP)}                  {_btn(Button.Y)}",
            f"     {_btn(Button.DPAD_LEFT)} + {_btn(Button.DPAD_RIGHT)}"
            f"    {_btn(Button.BACK)}  {_btn(Button.START)}"
            f"    {_btn(Button.X)} + {_btn(Button.A)}",
            f"       {_btn(Button.DPAD_DOWN)}                  {_btn(Button.B)}",
            "",
            f"   L {_stick(lx, ly)} {_btn(Button.L3)}"
            f"       R {_stick(rx, ry)} {_btn(Button.R3)}",
            "",
            f"   connected={self._connected}  updates={self._update_count}",
        ]
        print("\n".join(lines))

    # -- internals ----------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Gamepad is not connected")


# Alias for convenience
MockGamepad = MockGamepadBackend

# Structural check: ensure the class satisfies the protocol at import time.
assert isinstance(MockGamepadBackend(), GamepadBackend)  # type: ignore[arg-type]
