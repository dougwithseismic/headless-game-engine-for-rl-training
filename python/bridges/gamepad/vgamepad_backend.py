"""Real Xbox 360 virtual controller backend using vgamepad (ViGEmBus).

This module wraps the ``vgamepad`` library which creates a virtual Xbox 360
controller on Windows via the ViGEmBus driver.  It implements the same
:class:`GamepadBackend` protocol as :class:`MockGamepadBackend`.

On non-Windows platforms (or when ``vgamepad`` is not installed) the module
can still be imported -- it just raises :class:`RuntimeError` at
:meth:`VGamepadBackend.connect` time.
"""

from __future__ import annotations

from bridges.gamepad.protocol import (
    Axis,
    Button,
    DEFAULT_DEADZONE,
    GamepadBackend,
    GamepadState,
    STICK_AXES,
    apply_deadzone,
)

try:
    import vgamepad as vg

    _VGAMEPAD_AVAILABLE = True
except ImportError:
    vg = None  # type: ignore[assignment]
    _VGAMEPAD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Button name -> vgamepad XUSB_BUTTON constant
# ---------------------------------------------------------------------------

_BUTTON_MAP: dict[Button, str] = {
    Button.A: "XUSB_GAMEPAD_A",
    Button.B: "XUSB_GAMEPAD_B",
    Button.X: "XUSB_GAMEPAD_X",
    Button.Y: "XUSB_GAMEPAD_Y",
    Button.LB: "XUSB_GAMEPAD_LEFT_SHOULDER",
    Button.RB: "XUSB_GAMEPAD_RIGHT_SHOULDER",
    Button.START: "XUSB_GAMEPAD_START",
    Button.BACK: "XUSB_GAMEPAD_BACK",
    Button.L3: "XUSB_GAMEPAD_LEFT_THUMB",
    Button.R3: "XUSB_GAMEPAD_RIGHT_THUMB",
    Button.DPAD_UP: "XUSB_GAMEPAD_DPAD_UP",
    Button.DPAD_DOWN: "XUSB_GAMEPAD_DPAD_DOWN",
    Button.DPAD_LEFT: "XUSB_GAMEPAD_DPAD_LEFT",
    Button.DPAD_RIGHT: "XUSB_GAMEPAD_DPAD_RIGHT",
}


def _resolve_button(button: Button) -> object:
    """Resolve a :class:`Button` to the corresponding ``vg.XUSB_BUTTON`` value."""
    attr_name = _BUTTON_MAP.get(button)
    if attr_name is None:
        raise ValueError(f"Unmapped button: {button!r}")
    return getattr(vg.XUSB_BUTTON, attr_name)


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


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class VGamepadBackend:
    """Virtual Xbox 360 controller backed by ``vgamepad`` / ViGEmBus.

    Axis values are buffered internally because ``vgamepad`` sets both X and Y
    of a joystick in a single call.  Buffered values are flushed to the virtual
    device on each :meth:`update` call.
    """

    def __init__(self, *, deadzone: float = DEFAULT_DEADZONE) -> None:
        self._gamepad: object | None = None
        self._state = GamepadState()
        self._connected = False
        self.deadzone = deadzone

    # -- protocol methods --------------------------------------------------

    def connect(self) -> None:
        if self._connected:
            raise RuntimeError("Already connected")
        if not _VGAMEPAD_AVAILABLE:
            raise RuntimeError(
                "vgamepad is not installed or not available on this platform. "
                "Install it with `pip install vgamepad` on Windows with ViGEmBus."
            )
        self._gamepad = vg.VX360Gamepad()
        self._connected = True

    def disconnect(self) -> None:
        self._require_connected()
        # vgamepad doesn't expose an explicit teardown -- dropping the
        # reference lets the destructor clean up the ViGEm target.
        self._gamepad = None
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
        """Flush all buffered axis/button state to the virtual controller."""
        self._require_connected()
        gp = self._gamepad

        # -- sticks (both axes per stick in one call) ----------------------
        gp.left_joystick_float(
            x_value_float=self._state.axes[Axis.LEFT_STICK_X],
            y_value_float=self._state.axes[Axis.LEFT_STICK_Y],
        )
        gp.right_joystick_float(
            x_value_float=self._state.axes[Axis.RIGHT_STICK_X],
            y_value_float=self._state.axes[Axis.RIGHT_STICK_Y],
        )

        # -- triggers ------------------------------------------------------
        gp.left_trigger_float(
            value_float=self._state.axes[Axis.LEFT_TRIGGER],
        )
        gp.right_trigger_float(
            value_float=self._state.axes[Axis.RIGHT_TRIGGER],
        )

        # -- buttons -------------------------------------------------------
        for button, pressed in self._state.buttons.items():
            vg_button = _resolve_button(button)
            if pressed:
                gp.press_button(button=vg_button)
            else:
                gp.release_button(button=vg_button)

        # -- flush to driver -----------------------------------------------
        gp.update()

    def reset(self) -> None:
        self._require_connected()
        self._state.reset()
        self.update()

    # -- internals ---------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Gamepad is not connected")
