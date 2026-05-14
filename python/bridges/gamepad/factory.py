"""Platform-aware factory for virtual gamepad backends.

Provides :func:`make_gamepad` which returns the right backend for the current
OS, and :func:`make_pid_gamepad` which pairs a backend with a PID controller.
"""

from __future__ import annotations

import sys

from bridges.gamepad.mock_backend import MockGamepadBackend
from bridges.gamepad.pid import GamepadPIDController, PIDPreset
from bridges.gamepad.protocol import GamepadBackend

__all__ = [
    "GamepadBackend",
    "GamepadPIDController",
    "MockGamepadBackend",
    "PIDPreset",
    "make_gamepad",
    "make_pid_gamepad",
]


def make_gamepad(backend: str = "auto", **kwargs: object) -> GamepadBackend:
    """Create a virtual gamepad backend.

    Parameters
    ----------
    backend:
        ``"auto"`` -- Windows gets :class:`VGamepadBackend`, everything else
        gets :class:`MockGamepadBackend`.
        ``"vgamepad"`` -- Force the real ViGEmBus backend (raises if
        unavailable).
        ``"mock"`` -- Force the mock backend.
    **kwargs:
        Forwarded to the backend constructor (e.g. ``max_history`` for the
        mock).
    """
    if backend == "auto":
        if sys.platform == "win32":
            from bridges.gamepad.vgamepad_backend import VGamepadBackend

            return VGamepadBackend(**kwargs)  # type: ignore[arg-type]
        return MockGamepadBackend(**kwargs)  # type: ignore[arg-type]

    if backend == "vgamepad":
        if sys.platform != "win32":
            raise RuntimeError(
                f"vgamepad backend requires Windows, but platform is {sys.platform!r}"
            )
        from bridges.gamepad.vgamepad_backend import VGamepadBackend

        return VGamepadBackend(**kwargs)  # type: ignore[arg-type]

    if backend == "mock":
        return MockGamepadBackend(**kwargs)  # type: ignore[arg-type]

    raise ValueError(
        f"Unknown backend {backend!r}. Choose from: 'auto', 'vgamepad', 'mock'."
    )


def make_pid_gamepad(
    backend: str = "auto",
    presets: dict[str, PIDPreset] | None = None,
    dt: float | None = None,
    **kwargs: object,
) -> tuple[GamepadBackend, GamepadPIDController]:
    """Create a gamepad + PID controller pair, ready to use.

    Parameters
    ----------
    backend:
        Forwarded to :func:`make_gamepad`.
    presets:
        Per-axis PID gain overrides.  See :class:`GamepadPIDController`.
    dt:
        Fixed timestep for the PID controller (``None`` = wall-clock).
    **kwargs:
        Extra keyword arguments forwarded to :func:`make_gamepad`.

    Returns
    -------
    tuple[GamepadBackend, GamepadPIDController]
        The backend (call ``.connect()`` before use) and its paired PID
        controller.
    """
    pad = make_gamepad(backend=backend, **kwargs)
    pid = GamepadPIDController(presets=presets, dt=dt)
    return pad, pid
